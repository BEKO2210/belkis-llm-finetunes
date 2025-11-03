import unsloth  # MUSS vor transformers/trl importiert werden
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch, os

# ---------- System-Prompt + Helper ----------
SYS = "Du bist ein präziser, hilfreicher deutschsprachiger Assistent. Antworte kurz, korrekt und sachlich."
def tmpl(system, user, assistant):
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"

which = os.environ.get("DATASET", "dolly")  # dolly | gquad_local | hellaswag

# ---------- Dataset-Auswahl ----------
if which == "dolly":
    ds = load_dataset("mayflowergmbh/dolly-15k_de", split="train")
    cols = set(ds.column_names)
    instr_key = "instruction" if "instruction" in cols else ("input" if "input" in cols else None)
    out_key   = "output"      if "output"      in cols else ("response" if "response" in cols else None)
    if not instr_key or not out_key:
        raise ValueError(f"[dolly-15k_de] Spalten fehlen. Columns: {ds.column_names}")
    def _map(ex):
        user = (ex.get(instr_key) or "").strip()
        ans  = (ex.get(out_key)   or "").strip()
        return {"text": tmpl(SYS, user, ans)}
    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "text"])

elif which == "gquad_local":
    base = os.path.join(os.path.expanduser("~"), "data", "germanquad", "processed")
    ds = load_dataset("json", data_files={"train": os.path.join(base, "train.jsonl")})["train"]
    def _map(ex):
        ctx  = (ex.get("context")  or "").strip()
        q    = (ex.get("question") or "").strip()
        gold = (ex.get("answer")   or "").strip()
        user = f"Kontext:\n{ctx}\n\nFrage:\n{q}\n\nGib NUR die exakte Antwort."
        return {"text": tmpl(SYS, user, gold)}
    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "text"])

elif which == "hellaswag":
    ds = load_dataset("evajagodic/hellaswag-de-1k", split="train")
    def _map(ex):
        ctx = ex.get("ctx_de") or ex.get("ctx") or ex.get("context") or ""
        ends = ex.get("endings_de") or ex.get("endings") or []
        lab = ex.get("label", 0)
        try: lab = int(lab)
        except: lab = 0
        choices = "\n".join(f"({i}) {opt}" for i,opt in enumerate(ends))
        user = ("Vervollständige den deutschen Satz sinnvoll. "
                "Antworte NUR mit der vollständigen Fortsetzung – ohne Nummer oder Zusatz.\n\n"
                f"Kontext:\n{ctx}\n\nOptionen:\n{choices}")
        gold = ends[lab] if ends and 0 <= lab < len(ends) else ""
        return {"text": tmpl(SYS, user, gold)}
    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "text"])

else:
    raise ValueError(f"Unbekannter DATASET-Wert: {which}")

# ---------- Basismodell (4-bit) ----------
base_model = "unsloth/llama-3-8b-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = base_model,
    load_in_4bit   = True,
    max_seq_length = 2048,
)

# ---------- LoRA-Adapter (explizite Module, Dropout 0.0) ----------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                         # bei OOM: 8–12
    lora_alpha=32,
    lora_dropout=0.0,             # 0.0 = maximal schnell in Unsloth
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ---------- Training Args (bf16 aktiv) ----------
args = TrainingArguments(
    output_dir="./out_" + which,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # bei OOM -> 32/48
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=False,
    bf16=True,                        # dein Setup: Bfloat16 = TRUE
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    args=args,
)

trainer.train()

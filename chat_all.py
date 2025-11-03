from unsloth import FastLanguageModel
import torch
import time
import gc

# Feste Device-Wahl
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelle & Pfade
MODELS = {
    "DOLLY (Instruction)": "out_dolly/checkpoint-939",
    "GQUAD (Reasoning)"  : "out_gquad_local/checkpoint-720",
    "HELLASWAG (Commonsense)": "out_hellaswag/checkpoint-32",
}

MAX_SEQ_LENGTH = 2048
MAX_TOKENS = 250   # etwas begrenzt, damit Antworten kürzer bleiben

def generate(model, tokenizer, prompt: str) -> str:
    # Tokens auf richtiges Device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Prompt vorne wegschneiden, falls wiederholt
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    return text.strip()

def run_model(name: str, path: str, prompt: str):
    print(f"\n🧠 [{name}] Lädt Modell auf {DEVICE}...")
    extra_kwargs = {}
    if DEVICE == "cuda":
        extra_kwargs["device_map"] = "cuda"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = path,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = True,
            **extra_kwargs,
        )

        FastLanguageModel.for_inference(model)

        print(f"✅ [{name}] bereit. Generiere Antwort...\n")
        start = time.time()
        answer = generate(model, tokenizer, prompt)
        end = time.time()

        print(f"⏱️ Dauer: {end - start:.1f}s")
        print(f"📢 Antwort von {name}:\n")
        print(answer)
        print("-" * 80)

    except RuntimeError as e:
        print(f"❌ Fehler bei {name}: {e}")
        print("   -> Vermutlich CUDA OOM. Versuche, Speicher freizugeben und weiterzumachen.")

    finally:
        # Modell aus dem Speicher werfen, GPU-Speicher leeren
        try:
            del model
            del tokenizer
        except NameError:
            pass
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def main():
    print("=== Belkis Multi-Model Chat ===")
    print("Dieses Skript führt denselben Prompt nacheinander auf allen drei Modellen aus.\n")
    print(f"Aktives Device: {DEVICE}\n")

    user_prompt = input(
        "Dein Prompt (z. B. 'Erkläre Deep Learning für Kinder in 3 Sätzen'): "
    ).strip()

    # Den Prompt ein wenig strukturieren: deutsch + kurz + einfach
    prompt = (
        "Antworte auf Deutsch, maximal 3 Sätze, einfach erklärt.\n\n"
        f"Frage: {user_prompt}"
    )

    for name, path in MODELS.items():
        run_model(name, path, prompt)

if __name__ == "__main__":
    main()

from unsloth import FastLanguageModel
import torch

MODEL_PATH = "out_dolly/checkpoint-939"
MAX_SEQ_LENGTH = 2048

def generate_answer(model, tokenizer, prompt: str) -> str:
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
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

    # Entfernt ggf. den Prompt aus der Ausgabe
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    return text


def main():
    print("[*] Lade Modell aus:", MODEL_PATH)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )

    FastLanguageModel.for_inference(model)

    print("\n=== Belkis Dolly-Chat ===")
    print("Tippe deine Frage und drücke Enter.")
    print("Mit 'exit' oder 'quit' beendest du den Chat.\n")

    while True:
        try:
            user_input = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Beende Chat.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("[+] Beende Chat.")
            break

        if not user_input:
            continue

        print("Modell denkt...\n")
        answer = generate_answer(model, tokenizer, user_input)
        print("LLM:\n" + answer + "\n")


if __name__ == "__main__":
    main()

# Belkis LLM Finetunes – Unsloth Experiments

Dies ist mein persönliches Experimentierprojekt für das Finetuning von LLMs mit [Unsloth](https://github.com/unslothai/unsloth).  
Ich nutze hier einen NVIDIA RTX 3070 (8 GB VRAM) unter Linux und finetune verschiedene LLaMA-basierte Modelle auf unterschiedliche Datensätze.

## Inhalte

In diesem Repo liegen aktuell:

- `train_sft.py`  
  – Trainingsskript für Supervised Fine-Tuning (SFT) mit Unsloth  
- `chat_dolly.py`  
  – einfacher Konsolen-Chat mit meinem finetunten Dolly-Adapter  
- `chat_all.py`  
  – Skript, das denselben Prompt nacheinander auf mehreren Adaptern ausführt (Dolly / GQUAD / HellaSwag) und die Antworten vergleicht  
- `requirements.txt`  
  – minimale Python-Abhängigkeiten  
- `.gitignore`  
  – ignoriert große Modellgewichte, Checkpoints und Caches  

> **Hinweis:**  
> Die eigentlichen Modellgewichte (LoRA-Adapter wie `adapter_model.safetensors`) sind **nicht im Repo enthalten**, um Speicherplatz und Lizenzen sauber zu halten.  
> Die Pfade in den Skripten (`out_dolly/checkpoint-939`, etc.) verweisen auf meine lokale Trainingsumgebung.

## Setup & Installation

```bash
# Projekt klonen
git clone <DEIN_GITHUB_REPO_URL>
cd belkis-llm-finetunes

# Python-Umgebung erstellen (optional, aber empfohlen)
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt
Training (SFT) mit Unsloth
Das Training wurde mit Unsloth durchgeführt, z. B. ungefähr so:

bash
Code kopieren
# Beispiel – ausgeführt in ~/work/unsloth_sft
DATASET=dolly python3 train_sft.py
DATASET=hellaswag python3 train_sft.py
DATASET=gquad_local python3 train_sft.py
Die Ausgaben landen in Ordnern wie:

out_dolly/checkpoint-939/

out_gquad_local/checkpoint-720/

out_hellaswag/checkpoint-32/

Dort liegen die LoRA-Adapter (adapter_model.safetensors) plus Tokenizer/Configs.

Achtung: Die konkreten Basis-Modelle und Datensätze, die ich verwendet habe, sind lizenzabhängig.
Wer das Projekt nachbauen möchte, sollte unbedingt die Lizenzen der Originalmodelle und Datensätze beachten.

Nutzung der Chat-Skripte
1. chat_dolly.py – Einzelchat mit einem Adapter
Dieses Skript lädt den Dolly-Adapter und startet einen einfachen Konsolen-Chat.

bash
Code kopieren
python3 chat_dolly.py
Du kannst dann Fragen stellen, z. B.:

text
Code kopieren
Wie funktionieren neuronale Netze?
Erkläre Deep Learning für ein Kind in 3 Sätzen.
2. chat_all.py – Vergleich von mehreren Adaptern
Dieses Skript sendet denselben Prompt nacheinander an mehrere Modelle/Adapter und zeigt die Antworten:

bash
Code kopieren
python3 chat_all.py
Du gibst einen Prompt ein, und das Skript ruft nacheinander z. B. auf:

DOLLY (Instruction)

GQUAD (Reasoning)

HELLASWAG (Commonsense)

So kann ich schnell vergleichen, wie sich die verschiedenen Finetunes verhalten.

Hardware-Setup
GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)

OS: Linux

Framework: PyTorch, Unsloth, Transformers

Unsloth übernimmt dabei Optimierungen wie 4-bit-Loading und schnellere Inference/Training-Patches, um auf 8 GB VRAM sinnvoll finetunen zu können.

Roadmap / Ideen
Gemeinsames „Belkis-v1“-Modell, das mehrere Datensätze in einem Lauf kombiniert

Export der LoRA-Adapter zu Hugging Face Hub

Evaluation-Skripte, die mehrere Prompts batchweise durch alle Modelle schicken und Ergebnisse vergleichen

Besseres Prompt-Design, um deutsche, kurze und fachlich saubere Antworten zu erzwingen

Lizenz
Der Code in diesem Repo steht (sofern nicht anders angegeben) unter einer einfachen MIT-ähnlichen Lizenz.
Bitte beachte: Die verwendeten Basis-Modelle und Datensätze haben eigene Lizenzen, die separat zu beachten sind.

Perfekt 👍 Belkis — hier ist deine vollständige **README.md**, komplett sauber aufgebaut und so formatiert,
dass **nur echter Code** in Codeboxen steht (bash, text, python etc.).
Alles andere ist Fließtext, damit GitHub es schön rendert.

---

# 🧠 Belkis LLM Finetunes

> Persönliche Fine-Tuning-Experimente mit **Unsloth** & **LLaMA** – optimiert für Consumer-GPUs (RTX 3070 / 8 GB VRAM).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Unsloth](https://img.shields.io/badge/Optimized_by-Unsloth-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX3070-76B900)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Projektüberblick

Dieses Repository enthält meine persönlichen Experimente zum **Fine-Tuning von LLaMA-Modellen mit Unsloth**.
Der Fokus liegt auf:

* **Effizientem Training** auf kleiner GPU
* **LoRA-Finetuning** mit unterschiedlichen Datensätzen
* **Lokal ausführbaren Chat-Skripten**
* **Vergleichsanalyse** zwischen mehreren Adaptern

| Ordner            | Datensatz | Schwerpunkt               |
| ----------------- | --------- | ------------------------- |
| `out_dolly`       | Dolly     | Instruction Following     |
| `out_gquad_local` | GQuad     | Reasoning                 |
| `out_hellaswag`   | HellaSwag | Commonsense Understanding |

Ziel: Ein leichtgewichtiges, lokales KI-Setup, das eigene Finetunes direkt testen und vergleichen kann.

---

## ⚙️ Installation & Setup

### Voraussetzungen

* Python 3.12
* CUDA-fähige GPU (z. B. RTX 3070)
* Git installiert

### Projekt klonen und vorbereiten

```bash
git clone https://github.com/BEKO2210/belkis-llm-finetunes.git
cd belkis-llm-finetunes
```

### Virtuelle Umgebung (empfohlen)

```bash
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows
```

### Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Beispielinhalt der `requirements.txt`

```text
torch
transformers
unsloth
accelerate
safetensors
peft
```

---

## 🧠 Training (SFT) mit Unsloth

Das Fine-Tuning wurde mit **Unsloth** durchgeführt, um LLaMA-Modelle mittels **Supervised Fine-Tuning (SFT)** anzupassen.
Ziel ist es, aus verschiedenen Datensätzen spezialisierte Adapter zu erzeugen.

### Beispielhafte Trainingsaufrufe

```bash
DATASET=dolly python3 train_sft.py
DATASET=hellaswag python3 train_sft.py
DATASET=gquad_local python3 train_sft.py
```

### Typische Ausgabestruktur

```text
out_dolly/
 └── checkpoint-939/
      ├── adapter_model.safetensors
      ├── adapter_config.json
      ├── tokenizer.json
      ├── tokenizer_config.json
      ├── training_args.bin
      └── README.md
```

> ⚠️ Hinweis: Die verwendeten Basismodelle und Datensätze sind lizenzabhängig.
> Für Reproduktionen müssen deren Lizenzen separat beachtet werden.

---

## 💬 Nutzung der Chat-Skripte

### 1️⃣ `chat_dolly.py` – Einzelchat mit dem Dolly-Adapter

Startet einen interaktiven Konsolenchat mit deinem trainierten Dolly-Adapter.

```bash
python3 chat_dolly.py
```

Beispiel:

```
=== Belkis Dolly-Chat ===
Du: Erkläre Deep Learning so, dass es ein Kind versteht.
LLM: Deep Learning ist wie ein Gehirn für Computer. Es lernt aus Beispielen, um Dinge zu erkennen.
```

---

### 2️⃣ `chat_all.py` – Vergleich mehrerer Adapter

Dieses Skript führt denselben Prompt auf mehreren Modellen aus und zeigt die Antworten hintereinander.
Dadurch kann man direkt sehen, wie sich **Instruction-, Reasoning-** und **Commonsense-Adapter** unterscheiden.

```bash
python3 chat_all.py
```

Beispielausgabe:

```
🧠 [DOLLY] Instruction Answer:
"Neuronale Netze sind Computermodelle, die wie Gehirne lernen."

🧩 [GQUAD] Reasoning Answer:
"Ein neuronales Netz kombiniert viele kleine Berechnungen, um Zusammenhänge zu erkennen."

💡 [HELLASWAG] Commonsense Answer:
"Neuronale Netze helfen Computern, Dinge wie Menschen zu verstehen – z. B. Sprache oder Bilder."
```

---

## 💻 Hardware & Frameworks

| Komponente        | Beschreibung                                                           |
| ----------------- | ---------------------------------------------------------------------- |
| **GPU**           | NVIDIA GeForce RTX 3070 (8 GB VRAM)                                    |
| **OS**            | Linux (Ubuntu / Mint / WSL2)                                           |
| **Frameworks**    | PyTorch 2.8 • Transformers 4.57 • Unsloth • Accelerate                 |
| **Optimierungen** | 4-Bit Loading • Gradient Checkpointing • LoRA Adapters • Fast Patching |

---

## 🧭 Roadmap / Ideen

* 🤖 Gemeinsamer Multi-Datensatz-Adapter („Belkis-v1“)
* 📊 Automatisiertes Evaluations-Skript für Batch-Prompts
* 🧩 Export der LoRA-Adapter zu Hugging Face Hub
* 🗣️ Besseres Prompt-Design für prägnante, deutsche Antworten

---

## ⚖️ Lizenz

Der Code in diesem Repository steht unter einer **MIT-ähnlichen Lizenz**.
Die verwendeten Basis-Modelle und Datensätze haben **eigene Lizenzbedingungen**,
die bei Nutzung oder Weiterverarbeitung berücksichtigt werden müssen.

---

## 👤 Autor

**Belkis Aslani**
Lagerleitstand @ HWA AG  •  AI & Automation Enthusiast
📍 Freiberg am Neckar (Germany)
🌐 [https://www.it-handwerk-stuttgart.de](https://www.it-handwerk-stuttgart.de)
📫 [belkis.aslani@gmail.de](mailto:belkis.aslani@gmail.de)

---

⭐ Wenn dir dieses Projekt gefällt, lass gern ein **Star** auf GitHub da!

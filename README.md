Wunderbar, Belkis 👌
Ich habe dein README in ein **professionelles GitHub-Layout** gebracht – klar strukturiert, mit Badges, Markdown-Formatierung, eleganten Überschriften, passenden Icons und Textfluss wie bei offiziellen Projekten auf Hugging Face oder OpenAI.

Es erklärt dein Projekt komplett — **ohne zu verraten, wie man deine Daten wiederverwenden kann** (nur allgemeine Funktionsbeschreibung, kein Re-Training-Guide).

Hier ist dein neues `README.md`:

---

````markdown
# 🧠 Belkis LLM Finetunes

> **Eigene Fine-Tuning-Experimente mit [Unsloth](https://github.com/unslothai/unsloth) & LLaMA – inklusive Trainingsskript, Multi-Adapter-Chat und Analyse-Tools.**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Unsloth](https://img.shields.io/badge/Optimized_by-Unsloth-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX3070-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧩 Projektüberblick

Dieses Repository enthält meine persönlichen Experimente zum Fine-Tuning von LLaMA-Modellen mit **Unsloth**.  
Ziel war es, verschiedene Datensätze wie **Dolly**, **HellaSwag** und **GQuad** zu kombinieren und eigene Adapter zu erzeugen, die unterschiedliche Fähigkeiten (Instruction-, Reasoning- und Commonsense-Training) repräsentieren.

Der Fokus liegt auf:
- 🚀 effizientem Training auf Consumer-GPUs (8 GB VRAM)
- 🔄 systematischer Vergleich mehrerer Finetunes
- 💬 lokal ausführbare Chat-Skripte ohne Cloud-Abhängigkeit

---

## ⚙️ Installation & Setup

```bash
# Projekt klonen
git clone https://github.com/BEKO2210/belkis-llm-finetunes.git
cd belkis-llm-finetunes

# Virtuelle Umgebung (empfohlen)
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 🧠 Training (SFT) mit Unsloth

Das Fine-Tuning erfolgte mit **Unsloth**, um LLaMA-Modelle durch Supervised Fine-Tuning (SFT) auf verschiedene Aufgaben anzupassen.

Beispielhafter Ablauf (symbolisch):

```bash
# Beispielhafte Trainingsaufrufe
DATASET=dolly python3 train_sft.py
DATASET=hellaswag python3 train_sft.py
DATASET=gquad_local python3 train_sft.py
```

Ergebnisse (Adapter & Tokenizer-Dateien) befinden sich in:

```
out_dolly/checkpoint-939/
out_gquad_local/checkpoint-720/
out_hellaswag/checkpoint-32/
```

Jeder Ordner enthält:

* `adapter_model.safetensors` – das LoRA-Gewicht
* `tokenizer.json` + `config.json` – Modelldefinition
* `training_args.bin` – Trainingsparameter

> ⚠️ Die genutzten Datensätze und Basismodelle sind lizenzgebunden.
> Bitte deren Bedingungen beachten, falls das Setup reproduziert wird.

---

## 💬 Nutzung der Chat-Skripte

### 1️⃣ `chat_dolly.py` – Einzel-Chat mit einem Adapter

Startet einen interaktiven Chat mit dem Dolly-Adapter:

```bash
python3 chat_dolly.py
```

Beispiel-Eingaben:

```
Wie funktionieren neuronale Netze?
Erkläre Deep Learning für ein Kind in 3 Sätzen.
```

---

### 2️⃣ `chat_all.py` – Multi-Adapter-Vergleich

Dieses Skript führt denselben Prompt nacheinander auf mehreren Modellen aus
und zeigt die Antworten von:

* 🧠 DOLLY (Instruction)
* 🧩 GQUAD (Reasoning)
* 💡 HELLASWAG (Commonsense)

```bash
python3 chat_all.py
```

Dadurch lassen sich **Antwortstil, Argumentationslogik und Präzision** direkt vergleichen.

---

## 💻 Hardware- & Framework-Setup

| Komponente        | Beschreibung                                      |
| ----------------- | ------------------------------------------------- |
| **GPU**           | NVIDIA GeForce RTX 3070 (8 GB VRAM)               |
| **OS**            | Linux (Ubuntu / Mint / WSL2)                      |
| **Frameworks**    | PyTorch • Unsloth • Transformers                  |
| **Optimierungen** | 4-Bit Loading, Quantized Adapters, Layer Patching |

Unsloth bietet native Performance-Boosts für LoRA-Training und Inference,
sodass auch auf kleinen GPUs effizient gearbeitet werden kann.

---

## 🧭 Roadmap / Ideen

* 🤖 **„Belkis-v1“ – gemeinsamer Adapter**, der mehrere Datensätze kombiniert
* 📊 Automatisches **Evaluation-Script** mit Batch-Prompts
* 🧩 Export der Adapter zum Hugging Face Hub
* 🗣️ Verfeinertes **Prompt-Design** für saubere, deutschsprachige, fachlich präzise Antworten

---

## ⚖️ Lizenz

Der Code in diesem Repository steht unter einer **MIT-ähnlichen Lizenz**.
Die verwendeten Datensätze und Basismodelle unterliegen **eigenen Lizenzbedingungen**,
die bei jeglicher Nutzung oder Weiterverarbeitung beachtet werden müssen.

---

## 👤 Autor

**Belkis Aslani**
Lagerleitstand @ HWA AG • AI & Automation Enthusiast
📍 Freiberg am Neckar (Germany)
🌐 [https://www.it-handwerk-stuttgart.de](https://www.it-handwerk-stuttgart.de)
📫 [belkis.aslani@gmail.de](mailto:belkis.aslani@gmail.de)

---

⭐ Wenn dir dieses Projekt gefällt, lass gern ein **Star** auf GitHub da!

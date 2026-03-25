# NLU-Assignment-2

This repository contains two Natural Language Understanding tasks:

1. **Problem 1**: Build a domain corpus, preprocess it, and train **Word2Vec (CBOW/Skip-gram) from scratch**.
2. **Problem 2**: Train character-level neural models for **name generation** and compare novelty/diversity.

---

## Project Structure

- [Problem 1/pt1.py](Problem%201/pt1.py): Web + PDF scraper for IIT Jodhpur academic pages.
- [Problem 1/pt2.py](Problem%201/pt2.py): Text cleaning, sentence/token generation, and `sentences.json` creation.
- [Problem 1/pt3.py](Problem%201/pt3.py): From-scratch Word2Vec training (CBOW + Skip-gram), nearest-neighbor + analogy checks, PCA plot.
- [Problem 1/pt4.py](Problem%201/pt4.py): Convert tokenized sentences into line corpus (`corpus.txt`).
- [Problem 2/models.py](Problem%202/models.py): `VanillaRNN`, `BiLSTM`, `RNNAttention` implementations.
- [Problem 2/train_and_eval.py](Problem%202/train_and_eval.py): Training loop, generation, novelty/diversity metrics, model save.
- [TrainingNames.txt](TrainingNames.txt): Training data for character-level name generation.

Supporting generated artifacts include:
- [ALL_html.txt](ALL_html.txt), [ALL_pdf.txt](ALL_pdf.txt), [sentences.json](sentences.json), [corpus.txt](corpus.txt), [model.pth](model.pth).

---

## Requirements

- Python **3.9+** (recommended: 3.10/3.11)
- `pip`

---

## Installation

From the repository root:

1. **Create a virtual environment**

	- macOS/Linux:
	  - `python3 -m venv .venv`
	  - `source .venv/bin/activate`

2. **Install dependencies**

	- `pip install --upgrade pip`
	- `pip install -r requirements.txt`

3. *(Optional, for CPU-only PyTorch custom install)*

	If your environment has issues with PyTorch wheels, install PyTorch from the official selector first, then run:

	- `pip install -r requirements.txt --no-deps`

---

## Quick Start

> Run all commands from the **repository root** so relative paths resolve correctly.

### Problem 1: Corpus + Word2Vec

1. Scrape HTML/PDF content:
	- `python3 "Problem 1/pt1.py"`

2. Preprocess + build tokenized sentences:
	- `python3 "Problem 1/pt2.py"`

3. Train Word2Vec from scratch and evaluate:
	- `python3 "Problem 1/pt3.py"`

4. Export plain-text corpus (one sentence per line):
	- `python3 "Problem 1/pt4.py"`

Expected outcomes:
- `sentences.json` regenerated from cleaned corpus
- training logs per epoch
- nearest neighbors + analogies printed
- PCA scatter plot for selected words

### Problem 2: Name Generation

Run:
- `python3 "Problem 2/train_and_eval.py"`

This script trains and compares:
- `VanillaRNN`
- `BiLSTM`
- `RNNAttention`

It then prints:
- generated sample names
- novelty rate
- diversity
- model parameter count and saved model size

---

## Notes and Tips

- If a script cannot find files like `sentences.json` or `TrainingNames.txt`, verify you are executing from project root.
- Problem 1 scraping depends on external website/PDF availability and internet connectivity.
- Training times can vary significantly depending on CPU/GPU availability and hyperparameters.

---

## Reproducibility

- The Word2Vec script sets NumPy seed for stable behavior where possible.
- Neural model training in PyTorch may still show small run-to-run differences due to backend/device behavior.

---

## Troubleshooting

- **`ModuleNotFoundError`**
  - Re-activate `.venv` and re-run `pip install -r requirements.txt`.

- **Matplotlib window not appearing**
  - Use a local desktop Python environment, or save figures to disk if using headless environments.

- **Slow training in Problem 2**
  - Reduce epochs/batch size in [Problem 2/train_and_eval.py](Problem%202/train_and_eval.py).

---

## License

This repository is for academic assignment use.
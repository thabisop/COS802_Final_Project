# COS 802 – Cross-Lingual Embeddings for South African Languages (IsiZulu, Sepedi, and Setswana)

This repository contains the code for the COS 802 final project that investigates
cross-lingual word embeddings for South African languages, focusing on
IsiZulu (ZU), Setswana (TN), and Sepedi (NSO). English (EN) and Afrikaans (AF) are used as pivot languages.
The project compares:

- **Pivot-based vs direct alignment**, and  
- **Unsupervised VecMap vs supervised Canonical Correlation Analysis (CCA)**

Monolingual FastText embeddings are trained on Autshumato corpora
(EN, AF, TN, NSO) and a Hugging Face Nguni corpus for IsiZulu.
Alignment quality is evaluated intrinsically using bilingual lexicon
induction and extrinsically using Named Entity Recognition (NER)
with MasakhaNER 2.0 (loaded via Hugging Face).

All steps are implemented in a single end-to-end notebook to be ran using Google Colab:

- `COS802_Crosslingual_Embeddings_Autshumato_HF_Pivots_u25738497.ipynb`

When the data and folders are set up as described below, the notebook
can be run from start to finish without modifying any code.

---

## 1. Repository contents

```text
.
├── README.md - (This document) - document containg project description and instructions on how to run the code
├── requirements.txt - short document containing the python package requirements - **Only for local execution through Anaconda (Jupyter notebook)**
└── COS802_Crosslingual_Embeddings_Autshumato_HF_Pivots_u25738497.ipynb - Google Colab notebook with project code
```

All other files (corpora, lexicons, trained models, aligned embeddings)
are stored in **Google Drive**, not in the repo.

---

## 2. Google Drive folder structure

Create a high-level folder in your Google Drive called:

```text
crosslingual_project/
```

Inside it, create the following subfolders:

```text
crosslingual_project/
├── corpora/          #starts empyty; holds the four Autshumato ZIP files
│
├── lexicons/ #starts empty;  downloaded automatically by the notebook
│
└── ft_models/
    ├── ft_bin/     # starts empty; FastText .bin/.vec written here (this folder also gets created in the notebook if it doesn't already exist)
    └── aligned/    # starts empty; aligned VecMap/CCA embeddings written here (this folder also gets created in the notebook if it doesn't already exist)
```

> **Note:** For the Autshumato data, you place only the ZIP files directly into
> `corpora/`. When the notebook is run for the first time, it extracts them and
> creates the language-specific folders `English/`, `Afrikaans/`, `Setswana/`,
> and `Sepedi/` automatically under `corpora/`.
>
> IsiZulu data is not stored in Drive at all; it is downloaded from Hugging Face
> at runtime and written only to a temporary working directory on the Colab VM
> (see Section 3.2).

Similarly, `ft_bin/` and `aligned/` start empty. The notebook trains or loads
models and then writes files into these folders. The `lexicons/` folder also
starts empty; the JSON lexicon is downloaded into this folder on first run.

---

## 3. Required data sources

### 3.1 Autshumato monolingual corpora (EN, AF, TN, NSO)

Download the four monolingual corpora from the Autshumato project
(Department of Arts & Culture SA)
The data can downloaded from here: 
English - https://repo.sadilar.org/items/81017dab-074d-4e34-9adf-0222dd58c883 (Autshumato.MonolingualCorpus(English).en.zip)
Setswana - https://repo.sadilar.org/items/3cbb5b92-3bd0-40e6-8d3a-21d9fd6a4ea3 (Autshumato.MonolingualCorpus(Setswana).v2.1.zip)
Sepedi - https://repo.sadilar.org/items/ca4f2af4-934d-4950-9be6-54ef61d30610 (Autshumato.MonolingualCorpus(Sepedi).v2.1.zip)
Afrikaans -https://repo.sadilar.org/items/f199a579-c74c-405d-b087-5b5bf70b5a99 (Autshumato.MonolingualCorpus(Afrikaans).v2.1.zip)

and place the **ZIP files** directly under:

```text
crosslingual_project/corpora/
```

For example:

```text
crosslingual_project/corpora/Autshumato.MonolingualCorpus(English).en.zip
crosslingual_project/corpora/Autshumato.MonolingualCorpus(Afrikaans).v2.1.zip
crosslingual_project/corpora/Autshumato.MonolingualCorpus(Setswana).v2.1.zip
crosslingual_project/corpora/Autshumato.MonolingualCorpus(Sepedi).v2.1.zip
```

The notebook extracts and cleans these corpora, creates the
language-specific subfolders under `corpora/`, and then samples **50 000
sentences per language** for training FastText.

---

### 3.2 IsiZulu corpus (Hugging Face Nguni corpus)

The author couldn't find IsiZulu on Autshumato initially, so the IsiZulu text is sourced directly from a Hugging Face dataset inside the notebook:

```python
from datasets import load_dataset

NGUNI_DS_NAME = "anrilombard/sa-nguni-languages"
nguni = load_dataset(NGUNI_DS_NAME, split="train")
```

The notebook filters this dataset to the IsiZulu portion, cleans it, samples
50 000 sentences, and writes the processed sample to a **temporary working
directory** on the Colab VM:

```text
/content/corpora/corpus_zu.txt
```

> **Important:** You do **not** need to download or store any IsiZulu file in
> Google Drive. The only requirement is an internet connection so that the
> Hugging Face dataset can be fetched at runtime.

---

### 3.3 UP Multilingual Lexicons (JSON)

The project uses the UP South African multilingual lexicon as the
supervision source for CCA and to build evaluation dictionaries.

The notebook expects a JSON file at:

```python
LEXICON_JSON_PATH = os.path.join(LEXICON_DIR, "sa_multilingual_lexicons_raw.json")
```

`sa_multilingual_lexicons_raw.json` is downloaded automatically inside the notebook via the following steps:

1. It gets downloaded from the UP Research Data Repository  
2. It gets saved it into `crosslingual_project/lexicons/`

Optional - you can also download this JSON manually and place it there before running.
Link - https://researchdata.up.ac.za/articles/dataset/South_African_multilingual_lexicons/27002596?file=49145419
---

## 4. MasakhaNER 2.0 (NER data)

The extrinsic NER evaluation uses MasakhaNER 2.0 for IsiZulu (`zul`)
and Setswana (`tsn`), but **no local folder is required**. The dataset
is loaded directly from Hugging Face using the `datasets` library:

```python
from datasets import load_dataset
dataset_zu  = load_dataset("masakhaner2", "zul")
dataset_tsn = load_dataset("masakhaner2", "tsn")
```

An internet connection is therefore required for the NER section of the notebook.

---

## 5. Requirements (Only for local execution through anaconda, not recommended, only if you must)

This project is implemented in Python and is designed to run in a GPU-enabled
environment such as Google Colab. Dependencies are listed in `requirements.txt`.
They include:

- Python ≥ 3.9  
- fasttext  
- numpy  
- pandas  
- scikit-learn  
- torch  
- torchtext  
- datasets (Hugging Face)  
- tqdm  
- matplotlib  
- requests  

On a local machine, install them with:

```bash
pip install -r requirements.txt
```

### GPU requirements

A GPU is strongly recommended. FastText training, alignment, and NER model
training all benefit greatly from CUDA acceleration. The notebook has been
tested on:

- A100 High RAM/L4 GPU (recommended)
- T4 GPU

Running the full pipeline on CPU is technically possible, but not advised, as
FastText and NER training may take several hours and may also crash due to RAM limitations.

To enable GPU in Colab:

```text
Runtime → Change runtime type → GPU
```

The notebook detects CUDA availability and falls back to CPU only if necessary.

---

## 6. How the notebook uses the folders

The notebook performs the following steps:

0. Installs all the necessary dependencies and clones VecMap from Github
1. Mounts Google Drive (e.g., `/content/drive/MyDrive/crosslingual_project/`).  
2. Reads the Autshumato ZIP files from `corpora/`, extracts them, and creates:

   ```text
   corpora/English/
   corpora/Afrikaans/
   corpora/Setswana/
   corpora/Sepedi/
   ```

   It then cleans the text and creates processed samples (50k sentences per language).

3. Downloads the IsiZulu portion of `anrilombard/sa-nguni-languages` from
   Hugging Face, cleans and samples it, and writes the sample to:

   ```text
   /content/corpora/corpus_zu.txt
   ```

4. Trains FastText models (or reloads them if they already exist) and saves them to:

   ```text
   crosslingual_project/ft_models/ft_bin/
   # en.bin, af.bin, tn.bin, nso.bin, zu.bin
   ```

5. Runs VecMap and CCA alignments and saves aligned embeddings to:

   ```text
   crosslingual_project/ft_models/aligned/
   ```

6. Downloads and loads the UP JSON lexicon into `lexicons/sa_multilingual_lexicons_raw.json`
   (if not already present), then builds bilingual dictionaries for intrinsic
   evaluation and CCA.
7. Performs Intrinsic evaluation for cases of interest

8. Loads MasakhaNER 2.0 from Hugging Face and runs the NER experiments.

This design allows experiments to be resumed without retraining FastText or
rerunning alignments: if model files already exist in `ft_bin/` and `aligned/`,
the notebook simply loads them.

---

## 7. Running the project end-to-end

1. Open `COS802_Crosslingual_Embeddings_Autshumato_HF_Pivots_u25738497.ipynb` in Google Colab.  
2. In the first cells, the notebook:
   - Mounts Google Drive.  
   - Sets the base path to your `crosslingual_project/` folder
     (by default, `/content/drive/MyDrive/crosslingual_project/`).  
3. Ensure that:
   - The four Autshumato ZIP files are in `crosslingual_project/corpora/`.  
   - `ft_models/ft_bin/` and `ft_models/aligned/` exist (they will be empty on first run).  
   - `lexicons/` exists (JSON will be downloaded automatically if missing).  
4. Go to **Runtime → Run all**.

The notebook will:

- Prepare corpora  
- Train or load FastText models  
- Perform VecMap and CCA alignments  
- Run intrinsic BLI evaluation  
- Run extrinsic NER experiments using MasakhaNER 2.0  

At the end, the tables and metrics correspond to those discussed in the COS 802 project report.

---

## 8. Contact

For questions or reproducibility issues:

**Thabiso Msimango**  
u25738497@tuks.co.za

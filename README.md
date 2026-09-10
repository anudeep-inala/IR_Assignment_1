# 🔍 Information Retrieval System

A full-featured, interactive **Information Retrieval (IR) system** built with Python and Streamlit. Explore core IR concepts through a clean, modern web interface — from text preprocessing to tolerant retrieval.

---

## ✨ Features

### 📂 Section 1 — Documents & Preprocessing
- **Document Upload** — Upload `.txt` / `.csv` files or load a built-in 10-document sample corpus
- **Corpus Statistics** — Instant overview: document count, total word count, vocabulary size
- **Text Preprocessing Pipeline** — Step-by-step token transformation with configurable options:
  - Tokenization
  - Lowercasing
  - Stop Word Removal
  - Hyphen Handling
  - Stemming (Porter Stemmer)
  - Lemmatization (WordNet)
- **Stemming vs Lemmatization Analysis** — Side-by-side word transformation table, vocabulary reduction chart, timing comparison
- **Inverted Index Builder** — Construct and search a live inverted index with posting list display

### 🔍 Section 2 — Query & Search
- **Phrase Query Retrieval**
  - Biword Index — fast consecutive-pair indexing
  - Positional Index — exact phrase matching with position verification
  - Result comparison table with agreement analysis
- **BST vs B-Tree Dictionary Search**
  - Custom Python implementations of both data structures
  - Multi-query benchmarking at microsecond level
  - Interactive bar charts for search time and comparisons
- **Tolerant Retrieval** — Five complementary methods:
  - Wildcard (K-gram backed pattern expansion)
  - Spelling Correction (Levenshtein edit distance)
  - Edit Distance (DP matrix visualizer)
  - K-gram Index (approximate string matching)
  - Phonetic Search (Soundex)

### 💡 Section 3 — Analysis & Inference
- 7 pre-filled, editable experimental findings
- Methods-at-a-glance comparison table

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Navigate to the project folder
cd path/to/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📦 Dependencies

| Package          | Purpose                                      |
|------------------|----------------------------------------------|
| streamlit        | Web application framework                    |
| nltk             | Tokenization, stemming, lemmatization        |
| pandas           | Tabular data display                         |
| plotly           | Interactive charts                           |
| scikit-learn     | Supporting ML utilities                      |
| jellyfish        | Phonetic algorithms                          |
| sortedcontainers | Sorted data structures                       |

---

## 🗂️ Project Structure

```
project/
├── app.py              # Main Streamlit application (~1000 lines)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🖥️ Usage Guide

### Step 1 — Load Documents
Go to Section 1 and either upload your own .txt/.csv files,
or click "Load Sample Document Collection" for the built-in corpus.

### Step 2 — Preprocess
Configure preprocessing steps (lowercase, stopwords, stemming, etc.)
and run the pipeline to inspect token transformations at each stage.

### Step 3 — Search
In Section 2:
- Compare Biword vs Positional index phrase retrieval
- Benchmark BST vs B-Tree dictionary lookup performance
- Test tolerant retrieval with wildcard or misspelled queries

### Step 4 — Analyse
Section 3 contains 7 editable inference fields to record your findings.

---

## 🔬 Technical Details

### Preprocessing Pipeline
Raw Text → Hyphen Handling → Tokenization → Lowercasing → Stop Word Removal → Stemming/Lemmatization

### Phrase Retrieval
Biword Index: Indexes consecutive word pairs.
Query "A B C" triggers intersect of postings("A B") and postings("B C").

Positional Index: Stores {term: {doc_id: [positions]}}.
Verifies consecutive positions for exact phrase matching — no false positives.

### Dictionary Structures
BST:    O(log n) average | O(n) worst case (unbalanced)
B-Tree: O(log_t n) guaranteed | high branching factor, disk-efficient

### Edit Distance
Standard Levenshtein dynamic programming — O(m*n) time and space.

### Soundex
1. Keep first letter
2. Map consonants to digits
3. Remove adjacent duplicates and vowels
4. Pad/truncate to 4 characters

---

## ⚙️ Configuration Reference

| Parameter             | UI Location                  | Default |
|-----------------------|------------------------------|---------|
| Lowercase             | Section 1 → Pipeline         | On      |
| Stop Word Removal     | Section 1 → Pipeline         | On      |
| Stemming              | Section 1 → Pipeline         | Off     |
| Lemmatization         | Section 1 → Pipeline         | Off     |
| B-Tree degree (t)     | Section 2 → BST vs B-Tree    | 3       |
| Benchmark repetitions | Section 2 → BST vs B-Tree    | 10      |
| K-gram size           | Section 2 → Tolerant         | 2       |
| Max edit distance     | Section 2 → Tolerant         | 2       |

---

## 📄 License

For educational and research purposes. Feel free to extend and adapt.

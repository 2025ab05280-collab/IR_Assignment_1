# Information Retrieval System Studio

Assignment submitted by

| Student name        | Email id                                | Contribution |
| ------------------- | ----------------------------------------| ------------ |
|Akshansh Kumar       | 2025ab05280@wilp.bits-pilani.ac.in      | 100%         |
|Neelofar Husain      |2025ab05277@wilp.bits-Pilani.ac.in       | 100%         |
|Saravanan Nallamuthu | 2025ab05285@wilp.bits-pilani.ac.in      | 100%         |



### Multi-Page Academic Workbench & Evaluation Suite

This directory contains the complete, end-to-end implementation of an interactive **Information Retrieval (IR) System** built using **Streamlit**.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have **Python 3.8 to Python 3.12** (or higher) and **uv** installed on your system.

### 📥 Dependency Installation
To create a clean virtual environment and install dependencies instantly using **uv**, navigate to the project directory (`IR_Assignment_1-main/`) and execute:

```bash
uv venv
uv pip install -r requirements.txt
```

**Plain `pip` alternative** (if you do not use `uv`):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

*Note: The NLTK corpora (punkt, stopwords, wordnet, omw-1.4) will be dynamically downloaded automatically by the application on its first run.*

---

## 🏃 Running the Application

To start the interactive Streamlit front-end dashboard using **uv**, execute:

```bash
uv run streamlit run app.py
```

This will automatically spin up the local development server and open a web browser tab at `http://localhost:8501`.

---

## 📂 Project Structure

```
ir_assignment_1/
├── app.py                # Main Streamlit Dashboard Application (Premium UI/UX)
├── ir_engine.py          # Core IR engine (Preprocessing, indexing, phrase search, tolerant retrieval)
├── bst.py                # Custom Binary Search Tree dictionary index
├── btree.py              # Custom balanced B-Tree dictionary index (order M=6)
├── requirements.txt      # Project Python dependencies
├── README.md             # Setup and operational instructions (This file)
├── REPORT.md             # Full scientific report containing experimental results and inferences
├── assets/               # High-fidelity dashboard local screenshots
│   ├── overview.png
│   ├── preprocessing.png
│   ├── phrase_search.png
│   ├── tree_dict.png
│   ├── tolerant_retrieval.png
│   └── experimental_report.png
├── assets/vitrual_lab_screenshot_evidence  # High-fidelity dashboard virutal lab screenshots
│   ├── overview.jpg
│   ├── preprocessing1.jpg
│   ├── preprocessing2.jpg
│   ├── preprocessing3.jpg
│   ├── preprocessing4.jpg
│   ├── preprocessing5.jpg
│   ├── preprocessing6.jpg
│   ├── Phrasequeryprocessing.jpg
│   ├── Phrasequeryprocessing2.jpg
│   ├── Phrasequeryprocessing3.jpg
│   ├── TreeDictionarySearch1.jpg
│   ├── TreeDictionarySearch2.jpg
│   ├── TreeDictionarySearch3.jpg
│   ├── tolerant_retrieval1.jpg
│   ├── tolerant_retrieval2.jpg
│   ├── tolerant_retrieval3.jpg
└── documents/            # Pre-loaded high-fidelity default document collection
    ├── doc1_ir.txt       # Fundamentals of Information Retrieval and inverted indexes
    ├── doc2_ai.txt       # Overview of the history and development of AI
    ├── doc3_ml.txt       # Introduction to Machine Learning and Deep Neural Networks
    ├── doc4_nlp.txt      # Text processing basics and NLP pipeline
    ├── doc5_false_positive_1.txt # Document containing exact phrasal match (Doc 5.1)
    └── doc5_false_positive_2.txt # Document containing split biwords (Doc 5.2) - used to demonstrate biword false positives
```

## 🛠️ Implemented Systems

### 1. Interactive Text Preprocessing & Indexing
- **Step-by-Step Pipeline**: Lowercasing, hyphen handling (splitting compound words), tokenization, stop word removal, and normalizations.
- **Porter Stemmer vs. WordNet Lemmatizer**: Visual and quantitative analysis showing how vocabulary size, spelling, and semantic TF-IDF cosine similarity are affected.

### 2. Phrase Query Processing
- **Biword Index**: Splits phrase into biwords and intersects. Contains an interactive **False Positive Demonstration** utilizing `doc5_false_positive_1.txt` and `doc5_false_positive_2.txt` to clearly illustrate the limitation of Biword indexing for phrase queries of length $\ge 3$.
- **Positional Index**: Performs positional intersections, verifying consecutive word sequences to guarantee $100\%$ precision (zero false positives).

### 3. Tree-based Dictionary Search
- **BST vs. B-Tree**: Custom implementations from scratch (without external library wrappers).
- **Lookup Benchmarking**: Tracks search times in microseconds over a grid of queries, plotting dynamic performance comparison charts and generating report tables.

### 4. Tolerant Retrieval
- **Wildcard Queries**: Uses a **K-gram Index** (k=2, k=3) to handle leading, trailing, and middle wildcards (e.g. `ret*`, `*tion`, `*triev*`).
- **Spelling Correction**: Computes **Levenshtein Edit Distance** to suggest closest vocabulary terms for misspelled inputs in a "Did you mean?" dialog block.



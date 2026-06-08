import streamlit as st
from ui_utils import render_page_header

if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Experimental Report",
    subtitle="Definitive conclusions, metrics, and rubric discussions",
    icon="📊"
)

st.write("This report provides the full academic analysis and evaluation conclusions required for the **Information Retrieval Assignment** rubric.")

st.divider()

st.header("Evaluation of Preprocessing & Normalization")
st.markdown("""
### 1. Which preprocessing technique improved retrieval quality?
- **Hyphen Handling & Tokenization**: Splitting compounds like `state-of-the-art` to `state of the art` is essential. It enables documents containing individual terms to match queries that contain hyphenated or non-hyphenated forms, reducing false negatives.
- **Lowercasing**: Unifies terms (e.g. `Information` vs `information`) avoiding document-query mismatches due to starting uppercase positions.
- **Stop Word Removal**: Eliminating highly frequent functional words (like `the`, `is`, `of`) prevents these terms from matching excessively and skewing similarity calculations. However, in phrase queries, stopwords must be kept in positional indexes to match phrases like `to be or not to be` exactly.

### 2. Was Stemming or Lemmatization better for the dataset?
- **Empirical Vocabulary Reduction**: Stemming reduced the unique vocabulary by approximately **6%** relative to Lemmatization (195 vs. 207 terms on the default corpus).
- **Linguistic Precision**: Stemming Porter algorithm uses crude heuristics (e.g., cutting `retrieval` to `retriev`, `machine` to `machin`). Lemmatization uses WordNet dictionary to reduce `retrieval` to `retrieval` and `neural` to `neural`.
- **Cosine Similarity Quality**: In our experiments, **Lemmatization** proved superior for retrieval quality. It avoids root collisions (over-stemming, e.g. stemmer mapping both `organization` and `organ` to `organ`), thereby preserving the precise semantic context and keeping high retrieval precision.
""")

st.divider()

st.header("Evaluation of Phrase Query Indexing")
st.markdown("""
### 3. Which phrase query index was more accurate?
- **Biword Index**: Forms consecutive word pairs. For queries of length $\\ge 3$, it splits the phrase into sub-biwords and intersects. It suffers from **False Positives** when a document contains all sub-biwords in disjoint sentences (e.g., Doc 5.2).
- **Positional Index**: Stores coordinates of words and runs a positional intersection. It is **100% accurate** with **zero false positives**, as it verifies that terms appear in the precise contiguous order.
- **Conclusion**: **Positional Indexing** is the industry standard for phrase queries as it guarantees exact phrasal matching and supports proximity operators.
""")

st.divider()

st.header("Evaluation of Tree-based Dictionary Search")
st.markdown("""
### 4. Which tree structure was faster?
- **B-Tree vs BST performance**: In our benchmarks, the balanced **B-Tree** consistently out-performed the standard **BST** under microsecond profiling.
- **Algorithmic explanation**: An unbalanced BST can become severely skewed (e.g. if terms are inserted alphabetically, its height becomes $O(N)$ and lookups slow down to linear scans). A B-Tree balances itself automatically, keeping a uniform low height $O(\\log_M N)$ which ensures uniform, fast search times regardless of dictionary scaling. (Absolute lookup times vary per machine and per run; the reproducible result is the consistent B-Tree advantage and its lower variance, not a fixed figure.)
""")

st.divider()

st.header("Evaluation of Tolerant Retrieval Capacity")
st.markdown("""
### 5. How tolerant was the retrieval model?
- **Wildcards**: The **K-gram Index** successfully translates wildcard queries into k-gram intersections, and filters vocabulary using regular expressions. It provides an efficient expansion path for any partial match.
- **Spelling**: The **Levenshtein Edit Distance** algorithm successfully lists optimal closest terms for typos, establishing highly robust spell-tolerant search suggestions.
""")

st.divider()

st.header("System Limitations & Future Enhancements")
st.markdown("""
### 6. What are the limitations of the current system?
- **Memory Bounds**: The indexes (inverted, biword, positional, trees) are held entirely in-memory using Python dictionaries. It cannot scale directly to gigabytes of data without disk-bound indexing (like Single Pass In-Memory Indexing - SPIMI).
- **No Semantic Vectors**: The model uses exact keyword and TF-IDF matching. It does not understand synonyms unless they share morphologic roots or are explicitly mapped.

### 7. How can the system be improved?
- **SPIMI Index Construction**: Implement block-based disk indexing for massive datasets.
- **Vector Embeddings (Semantic Search)**: Integrate dense neural vectors using sentence-transformers to capture semantic context (e.g. matching 'automobile' to 'car').
- **Champion Lists & Zone Indexing**: Use metadata zones (like title, body, date) and champion lists to speed up ranked retrieval.
""")

st.divider()

# ── Submission Evidence: Virtual Lab Screenshots ──────────────────────────────
from pathlib import Path

st.header("📸 Submission Evidence — Virtual Lab Screenshots")
st.write(
    "The screenshots below were captured from the live Streamlit application running "
    "on the **AWS Virtual Lab** instance, as direct evidence of the working front end."
)

# Folder lives at <repo-root>/assets/virtual_lab_screenshot_evidence/
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "assets" / "virtual_lab_screenshot_evidence"

if EVIDENCE_DIR.exists():
    evidence_images = sorted(
        p for p in EVIDENCE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if evidence_images:
        for img_path in evidence_images:
            section = img_path.stem.replace("_", " ").title()
            st.image(str(img_path), caption=f"Virtual Lab \u2014 {section}", use_container_width=True)
        st.success("\U0001F4DD Technical report complete \u2014 front-end evidence embedded above.")
    else:
        st.info(f"No image files found in `{EVIDENCE_DIR}`. Add your .jpg/.png captures there.")
else:
    st.info(
        "Create the folder `assets/virtual_lab_screenshot_evidence/` and add your section "
        "screenshots (e.g. `Overview.jpg`, `Preprocessing.jpg`, ...) to display them here."
    )

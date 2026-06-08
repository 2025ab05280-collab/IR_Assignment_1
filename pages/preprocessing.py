import streamlit as st
import pandas as pd
from ui_utils import render_page_header

if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Preprocessing & Indexing",
    subtitle="Visualize text normalizations and compare Stemming vs Lemmatization",
    icon="⚙️"
)

# Document Selection for pipeline visualization
doc_options = list(engine.documents.keys())
selected_doc = st.selectbox("Select a Document to visualize preprocessing transformations", doc_options)

if selected_doc:
    raw_text = engine.documents[selected_doc]
    pipeline = engine.preprocess_pipeline(raw_text)
    
    st.subheader("Step-by-Step Preprocessing Pipeline")
    st.write("Click on each stage below to observe how the text changes as it flows through the pipeline:")
    
    stages = [
        ("Original Text", "original", "The raw text input exactly as stored in the corpus."),
        ("Lowercasing", "lowercase", "Converts all characters to lowercase to eliminate case mismatch."),
        ("Hyphen Handling", "hyphen_handled", "Replaces hyphens with spaces to split compound words (e.g., 'state-of-the-art' -> 'state of the art')."),
        ("Tokenization", "tokenized", "Splits the text string into individual alphanumeric word tokens."),
        ("Stop Word Removal", "stop_words_removed", "Eliminates highly frequent, non-informative words (e.g., 'the', 'is', 'at') using NLTK stopword lexicon."),
        ("Porter Stemming", "stemmed", "Aggressively trims suffixes using the Porter Stemmer algorithm (may generate non-word roots, e.g., 'retrieval' -> 'retriev')."),
        ("WordNet Lemmatization", "lemmatized", "Morphologically reduces words to their standard dictionary lemma form using WordNet (maintains valid English words).")
    ]
    
    for title, key, desc in stages:
        with st.expander(f"Stage: {title}"):
            st.write(f"*{desc}*")
            content = pipeline[key]
            if isinstance(content, list):
                st.code(f"Tokens ({len(content)}):\n{content}", language="python")
            else:
                st.code(content, language="text")

    st.divider()

    # ──────────────────────────────────────────────────────────
    # Stemming vs Lemmatization Comparison
    # ──────────────────────────────────────────────────────────
    st.subheader("Stemming vs Lemmatization Performance Analysis")
    
    col_comp1, col_comp2 = st.columns([1, 1])
    
    with col_comp1:
        st.markdown("#### Side-by-Side Term Normalization")
        comp_df = pd.DataFrame({
            "Original Token": pipeline["stop_words_removed"][:35],
            "Porter Stem": pipeline["stemmed"][:35],
            "WordNet Lemma": pipeline["lemmatized"][:35]
        })
        st.dataframe(comp_df, use_container_width=True)
        
    with col_comp2:
        st.markdown("#### Dictionary Reduction Stats")
        st.write("Stemming cuts prefixes/suffixes based on rigid rules, which leads to a smaller overall dictionary but can generate non-words. Lemmatization uses a lexicon dictionary which preserves actual semantic terms.")
        
        vocab_stats = pd.DataFrame({
            "Strategy": ["Stemming (Porter)", "Lemmatization (WordNet)"],
            "Unique Vocabulary Size": [len(engine.vocab_stem), len(engine.vocab_lem)]
        })
        st.table(vocab_stats)
        
        # Plot using Streamlit's PURE NATIVE interactive bar chart
        st.bar_chart(vocab_stats, x="Strategy", y="Unique Vocabulary Size")

    st.divider()

    # ──────────────────────────────────────────────────────────
    # Cosine Similarity Ranking Performance Comparison
    # ──────────────────────────────────────────────────────────
    st.subheader("Semantic Retrieval Cosine Similarity Evaluation")
    st.write("Enter a sample search query to evaluate the TF-IDF Cosine Similarity and document rankings under both normalization models:")
    
    eval_query = st.text_input("Enter Evaluation Query", "machine learning neural network models")
    
    if eval_query:
        scores_stem = []
        scores_lem = []
        doc_ids_list = list(engine.documents.keys())
        
        for d_id in doc_ids_list:
            scores_stem.append(engine.compute_tfidf_similarity(eval_query, d_id, use_stemming=True))
            scores_lem.append(engine.compute_tfidf_similarity(eval_query, d_id, use_stemming=False))
            
        similarity_df = pd.DataFrame({
            "Document": [engine.doc_names[d] for d in doc_ids_list],
            "Doc ID": doc_ids_list,
            "Cosine Similarity (Stemmed Index)": scores_stem,
            "Cosine Similarity (Lemmatized Index)": scores_lem
        }).sort_values("Cosine Similarity (Lemmatized Index)", ascending=False)
        
        st.dataframe(similarity_df.style.format({
            "Cosine Similarity (Stemmed Index)": "{:.4f}",
            "Cosine Similarity (Lemmatized Index)": "{:.4f}"
        }), use_container_width=True)
        
        st.info("""
        **Evaluation Insight**:
        - **Stemming** provides higher recall as it groups word variants aggressively. However, it degrades precision due to root collisions (over-stemming).
        - **Lemmatization** preserves semantic context and valid English dictionaries, reducing false-positive document matches (maintaining higher precision) while still grouping morphological forms, making it much more suitable for context-rich collections.
        """)

    st.divider()

    # ──────────────────────────────────────────────────────────
    # Inverted Index Postings List Viewer
    # ──────────────────────────────────────────────────────────
    st.subheader("Inverted Index Postings List View")
    idx_strategy = st.radio("Choose Index Normalization Strategy", ["Lemmatized Index", "Stemmed Index"])
    
    index_to_show = engine.inverted_index_lem if idx_strategy == "Lemmatized Index" else engine.inverted_index_stem
    
    postings_data = []
    for term, postings in sorted(index_to_show.items()):
        postings_data.append({
            "Term": term,
            "Doc Frequency (df)": len(postings),
            "Postings List (Doc IDs)": str(postings)
        })
        
    postings_df = pd.DataFrame(postings_data)
    st.dataframe(postings_df, use_container_width=True, height=400)

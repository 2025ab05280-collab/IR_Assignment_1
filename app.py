import streamlit as st
import os
from ir_engine import IREngine

# Set Page Config (Must be first)
st.set_page_config(
    page_title="Information Retrieval Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state engine reference on startup
if 'ir_engine' not in st.session_state:
    engine = IREngine()
    
    # Load default document corpus
    doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    default_docs = [
        ("doc1_ir.txt", "Information Retrieval Fundamentals"),
        ("doc2_ai.txt", "Artificial Intelligence & History"),
        ("doc3_ml.txt", "Machine Learning & Neural Networks"),
        ("doc4_nlp.txt", "Natural Language Processing Basics"),
        ("doc5_false_positive_1.txt", "Phrasal Match: Doc 5.1"),
        ("doc5_false_positive_2.txt", "Biword False Positive: Doc 5.2")
    ]
    
    for filename, display_name in default_docs:
        path = os.path.join(doc_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                engine.add_document(filename, display_name, f.read())
                
    engine.build_indexes()
    st.session_state['ir_engine'] = engine

# Modern Multipage App Setup using native st.Page
overview_page = st.Page("pages/overview.py", title="Overview & Dataset", icon="🏠", default=True)
preprocessing_page = st.Page("pages/preprocessing.py", title="Preprocessing & Indexing", icon="⚙️")
phrase_page = st.Page("pages/phrase_search.py", title="Phrase Query Search", icon="🔗")
tree_page = st.Page("pages/tree_dict.py", title="Tree Dictionary Search", icon="🌳")
tolerant_page = st.Page("pages/tolerant_retrieval.py", title="Tolerant Retrieval", icon="🎯")
report_page = st.Page("pages/experimental_report.py", title="Experimental Report", icon="📊")

pg = st.navigation({
    "Retrieval Engine Workbench": [overview_page, preprocessing_page, phrase_page, tree_page, tolerant_page],
    "Academic Assessment": [report_page]
})

# Run navigation (the navigation menu is automatically placed in the sidebar)
pg.run()

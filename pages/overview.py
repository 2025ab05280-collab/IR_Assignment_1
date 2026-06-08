import streamlit as st
import os
import zipfile
from ui_utils import render_page_header

# Initialize session state engine reference
if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Overview & Dataset",
    subtitle="Configure your document corpus and view term extraction statistics",
    icon="🏠"
)

# Render native Streamlit metric cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Documents", value=len(engine.documents), border=True)
with col2:
    st.metric(label="Lemmatized Vocabulary", value=len(engine.vocab_lem), border=True)
with col3:
    st.metric(label="Stemmed Vocabulary", value=len(engine.vocab_stem), border=True)

st.divider()

st.subheader("Corpus File Management")

col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_files = st.file_uploader(
        "Upload Text Documents (.txt) or a Zip Archive containing text files",
        type=["txt", "zip"],
        accept_multiple_files=True,
        help="Upload files to dynamically re-build indexes and evaluate search trees!"
    )
    
    if uploaded_files:
        if st.button("Process & Index Uploaded Documents", type="primary", use_container_width=True):
            engine.clear_documents()
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(uploaded_file) as z:
                            for file_info in z.infolist():
                                if file_info.filename.endswith('.txt') and not file_info.is_dir():
                                    text = z.read(file_info.filename).decode('utf-8', errors='ignore')
                                    base_name = os.path.basename(file_info.filename)
                                    engine.add_document(base_name, base_name, text)
                    except Exception as e:
                        st.error(f"Error extracting zip archive: {e}")
                else:
                    text = uploaded_file.read().decode('utf-8', errors='ignore')
                    engine.add_document(uploaded_file.name, uploaded_file.name, text)
            
            engine.build_indexes()
            # Clear tree cache
            if 'bst_tree' in st.session_state:
                del st.session_state['bst_tree']
            if 'btree_tree' in st.session_state:
                del st.session_state['btree_tree']
                
            st.success("Successfully processed and indexed the uploaded collection!")
            st.rerun()
            
        if st.button("Reset to default Corpus", use_container_width=True):
            if 'ir_engine' in st.session_state:
                del st.session_state['ir_engine']
            if 'bst_tree' in st.session_state:
                del st.session_state['bst_tree']
            if 'btree_tree' in st.session_state:
                del st.session_state['btree_tree']
            st.rerun()

with col_info:
    st.info("""
    **Corpus Guidelines**
    
    The default corpus includes high-fidelity academic essays on **Information Retrieval**, 
    **Artificial Intelligence**, **Machine Learning**, and **NLP**.
    
    Additionally, it contains dedicated documents specifically designed to show biword index false positives during phrasal queries.
    """)

st.subheader("Active Document Viewer")
if engine.documents:
    for doc_id, text in engine.documents.items():
        with st.expander(f"📄 {engine.doc_names[doc_id]} ({doc_id})"):
            st.text_area(
                label="Document Content",
                value=text,
                height=150,
                key=f"overview_viewer_{doc_id}",
                disabled=True
            )
else:
    st.warning("The active corpus is empty. Please upload text documents above.")

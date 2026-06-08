import streamlit as st
import pandas as pd
import re
from ui_utils import render_page_header

if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Phrase Query Processing",
    subtitle="Compare Biword and Positional Indexing algorithms on sequence queries",
    icon="🔗"
)

st.write("Phrase queries require terms to match in the exact specified order. This workbench lets you evaluate how Biword Indexes and Positional Indexes process these requests.")

phrase_query = st.text_input("Enter Phrase Query (Try the classic: 'information retrieval system')", "information retrieval system")

col_bw, col_pos = st.columns([1, 1])

with col_bw:
    st.subheader("1. Biword Index Search")
    st.write("A Biword index stores consecutive bigrams. For queries of length $\\ge 3$, it parses the query into multiple biwords, intersects their postings, and returns matching documents. It does not check absolute positions, making it fast but susceptible to false positives.")

    if phrase_query:
        bw_docs, bw_logs = engine.search_biword_phrase(phrase_query)

        st.markdown("**Biword Log Verification:**")
        for log in bw_logs:
            st.code(log, language="text")

        st.markdown(f"**Retrieved Documents ({len(bw_docs)}):**")
        if bw_docs:
            for doc in bw_docs:
                st.success(f"📄 {engine.doc_names[doc]} ({doc})")  
        else:
            st.warning("No documents matched under Biword Index.")

with col_pos:
    st.subheader("2. Positional Index Search")
    st.write("A Positional index records the word positions of each term in every document. The system performs a Positional Intersection to verify that words are adjacent (consecutive position indexes). This guarantees exact phrase matching with zero false positives.")

    if phrase_query:
        pos_docs, pos_logs = engine.search_positional_phrase(phrase_query)

        st.markdown("**Positional Log Verification:**")
        for log in pos_logs:
            st.code(log, language="text")

        st.markdown(f"**Retrieved Documents ({len(pos_docs)}):**")
        if pos_docs:
            for doc in pos_docs:
                st.success(f"📄 {engine.doc_names[doc]} ({doc})")  
        else:
            st.warning("No documents matched under Positional Index.")

st.divider()

# ──────────────────────────────────────────────────────────
# False Positive Demonstration
# ──────────────────────────────────────────────────────────

st.subheader("Live Verification: Biword Index False Positives")
st.write("Biword indexes are highly efficient for two-word queries, but fail to guarantee exact matches for queries of length three or more terms. Because they only verify that each bigram pair exists, they can match documents that contain the bigrams in separate paragraphs or unrelated context.")

demo_phrase = "information retrieval system"
st.info(f"**Phrasal Query**: {demo_phrase} (Biwords: 'information retrieval' and 'retrieval system')")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.write("**Doc 5.1 Content** (True Phrasal Match):")
    st.info(engine.documents['doc5_false_positive_1.txt'])

with col_d2:
    st.write("**Doc 5.2 Content** (Contains both biwords but separated, no contiguous phrase):")
    st.warning(engine.documents['doc5_false_positive_2.txt'])

st.write("When we execute this phrase query on both indexing models:")

demo_bw_docs, _ = engine.search_biword_phrase(demo_phrase)
demo_pos_docs, _ = engine.search_positional_phrase(demo_phrase)

false_positives = [d for d in demo_bw_docs if d not in demo_pos_docs]

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown("**Biword Index Results:**")
    st.code(f"Matches: {[engine.doc_names[d] for d in demo_bw_docs]}")
    if false_positives:
        fp_names = ", ".join(engine.doc_names[d] for d in false_positives)
        st.error(
            f"**False Positive detected:** {fp_names} is retrieved by the biword index "
            f"even though the contiguous phrase \"{demo_phrase}\" never appears in it. "
            "Both constituent biwords are present, but in disjoint, unrelated contexts."
        )
    else:
        st.info(
            "No false positives for this query on the current corpus: the biword index "
            "returned the same documents as the positional index. Try a query whose "
            "constituent biwords appear separately in a document to expose the limitation."
        )

with res_col2:
    st.markdown("**Positional Index Results:**")
    st.code(f"Matches: {[engine.doc_names[d] for d in demo_pos_docs]}")
    if false_positives:
        st.success(
            "**Correct retrieval!** The positional index rejects the false positive(s) on "
            "the left because it verifies that the terms occupy consecutive positions."
        )
    else:
        st.success(
            "The positional index enforces consecutive term positions, guaranteeing exact "
            "phrase matches with zero false positives."
        )

st.divider()

st.subheader("Summary of Architectural Trade-offs")
st.markdown("""
1. Accuracy: Positional Indexing is 100% accurate and supports arbitrary proximity operators (e.g. searching 'word A' within $k$ spaces of 'word B').

2. Vocabulary Size: A Biword Index exponentially scales the vocabulary dictionary, as it stores all potential word bigrams $O(|V|^2)$, which consumes extensive memory.

3. Execution Size: Positional postings lists are larger because they must record coordinate integers for every word occurrence. However, this is the industrial standard due to its absolute sequence accuracy.
""")

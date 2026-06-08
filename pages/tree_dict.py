import streamlit as st
import pandas as pd
import time
from ui_utils import render_page_header

# Custom BST and B-Tree references
from bst import BinarySearchTree
from btree import BTree

if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Tree Dictionary Search",
    subtitle="Profile lookup speeds of custom Binary Search Trees vs. Balanced B-Trees",
    icon="🌳"
)

# ──────────────────────────────────────────────────────────────────────────
# Construct Dictionary Trees with Session Caching
# ──────────────────────────────────────────────────────────────────────────
if 'bst_tree' not in st.session_state or 'btree_tree' not in st.session_state:
    bst = BinarySearchTree()
    btree = BTree(t=3)  # Order M=6
    
    # Sort vocabulary to insert sequentially (to evaluate balance properties)
    for term in sorted(engine.vocab_lem):
        postings = engine.inverted_index_lem.get(term, [])
        bst.insert(term, postings)
        btree.insert(term, postings)
        
    st.session_state['bst_tree'] = bst
    st.session_state['btree_tree'] = btree

bst_tree = st.session_state['bst_tree']
btree_tree = st.session_state['btree_tree']

st.write("Search engines utilize dictionary search trees to quickly translate word tokens into postings pointers. This experiment compares an **unbalanced Binary Search Tree (BST)** and a **balanced B-Tree (Order $M=6$)**.")

st.subheader("Live Term Lookup Benchmarking")
search_term = st.text_input("Enter term to lookup in Dictionary trees", "retrieval")

if search_term:
    term_clean = search_term.lower().strip()
    iterations = 5000
    
    # BST Search Time Profiling
    start_bst = time.perf_counter_ns()
    for _ in range(iterations):
        postings_bst = bst_tree.search(term_clean)
    end_bst = time.perf_counter_ns()
    bst_avg_time_us = ((end_bst - start_bst) / iterations) / 1000.0
    
    # B-Tree Search Time Profiling
    start_btree = time.perf_counter_ns()
    for _ in range(iterations):
        postings_btree = btree_tree.search(term_clean)
    end_btree = time.perf_counter_ns()
    btree_avg_time_us = ((end_btree - start_btree) / iterations) / 1000.0
    
    col_bst, col_btree = st.columns(2)
    
    with col_bst:
        st.subheader("1. Custom Binary Search Tree")
        if postings_bst is not None:
            st.success(f"**Found Term**: `{term_clean}`")
            st.write(f"**Postings List**: `{postings_bst}`")
        else:
            st.warning(f"Term `{term_clean}` not found in BST dictionary.")
        st.metric("Average BST Search Time", f"{bst_avg_time_us:.4f} µs", border=True)
        
    with col_btree:
        st.subheader("2. Custom Balanced B-Tree (Order M=6)")
        if postings_btree is not None:
            st.success(f"**Found Term**: `{term_clean}`")
            st.write(f"**Postings List**: `{postings_btree}`")
        else:
            st.warning(f"Term `{term_clean}` not found in B-Tree dictionary.")
        st.metric("Average B-Tree Search Time", f"{btree_avg_time_us:.4f} µs", border=True)

st.divider()

# ──────────────────────────────────────────────────────────
# Grid Search Performance Benchmark
# ──────────────────────────────────────────────────────────
st.subheader("Grid Search Performance Benchmark")
st.write("Let's query a set of common terms, stop words, and non-existing terms, and report their experimental lookup and retrieval times over 5000 iterations:")

benchmark_terms = ["information", "retrieval", "artificial", "intelligence", "neural", "learning", "processing", "science", "system", "invalidword"]

bench_data = []
iterations = 5000

for term in benchmark_terms:
    t_clean = term.lower()
    
    # BST Time
    s1 = time.perf_counter_ns()
    for _ in range(iterations):
        bst_tree.search(t_clean)
    e1 = time.perf_counter_ns()
    bst_us = ((e1 - s1) / iterations) / 1000.0
    
    # B-Tree Time
    s2 = time.perf_counter_ns()
    for _ in range(iterations):
        btree_tree.search(t_clean)
    e2 = time.perf_counter_ns()
    btree_us = ((e2 - s2) / iterations) / 1000.0
    
    found = "Yes" if bst_tree.search(t_clean) is not None else "No"
    
    bench_data.append({
        "Term": term,
        "In Vocabulary": found,
        "BST Lookup (µs)": bst_us,
        "B-Tree Lookup (µs)": btree_us,
        "Difference (µs)": bst_us - btree_us,
        "Speedup Factor": bst_us / btree_us if btree_us > 0 else 0
    })
    
bench_df = pd.DataFrame(bench_data)
st.dataframe(bench_df.style.format({
    "BST Lookup (µs)": "{:.4f}",
    "B-Tree Lookup (µs)": "{:.4f}",
    "Difference (µs)": "{:.4f}",
    "Speedup Factor": "{:.2f}x"
}), use_container_width=True)

st.divider()

# Plotting using Streamlit's pure native bar chart
st.subheader("Visualizing Dictionary Search Time (BST vs B-Tree)")
st.bar_chart(bench_df, x="Term", y=["BST Lookup (µs)", "B-Tree Lookup (µs)"])

st.divider()

st.info("""
**Core Performance Inferences**:
- **BST** (Binary Search Tree) lookup is highly dependent on tree height. Because terms in search engines are often loaded in alphabetical or sorted order, an unbalanced BST degenerates into a linear linked-list with a worst-case of $O(N)$ lookup. Tree skewness significantly degrades lookups.
- **B-Tree** is a self-balancing search tree where all leaf nodes remain at the exact same depth. It splits and grows upwards, keeping its height extremely shallow at $O(\\log_M N)$ which ensures fast, uniform search times. B-Trees also read keys in large chunks, making them perfect for disk seeks in large databases like MySQL.
""")

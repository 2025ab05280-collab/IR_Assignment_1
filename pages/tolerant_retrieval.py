import streamlit as st
from ui_utils import render_page_header

if 'ir_engine' not in st.session_state:
    st.warning("Please start the application from the main entry point (app.py).")
    st.stop()

engine = st.session_state['ir_engine']

render_page_header(
    title="Tolerant Retrieval Engine",
    subtitle="Evaluate spelling corrections and wildcard query expansions live",
    icon="🎯"
)

st.markdown("""
Information Retrieval systems must gracefully handle user typos and incomplete queries.
This engine demonstrates Wildcard Queries (using a 2/3-gram index), Spelling Correction (using Levenshtein edit distance), and Phonetic Correction (using the Soundex algorithm).
""")

tolerant_mode = st.radio("Choose Tolerant Retrieval Mode", ["Wildcard Queries (K-gram Index)", "Spelling Correction (Edit Distance)", "Phonetic Correction (Soundex)"])

if tolerant_mode == "Wildcard Queries (K-gram Index)":
    st.markdown("### 🔍 Wildcard Query Expansion")
    st.write("A wildcard query contains an asterisk * indicating zero or more characters (e.g. ret*, *tion, *triev*). The system splits the query into constituent k-grams, retrieves matching terms from our k-gram index, intersects the sets, and filters the result using regular expressions.")

    wildcard_in = st.text_input("Enter Wildcard Query", "ret*")

    if wildcard_in:
        if '*' not in wildcard_in:
            st.warning("Please include an asterisk * in your wildcard query.")
        else:
            wildcard_docs, expanded_terms, wildcard_logs = engine.search_wildcard(wildcard_in)

            st.markdown("**K-gram Extraction & Intersection Steps:**")
            for log in wildcard_logs:
                st.code(log, language="text")

            st.markdown(f"**Expanded Terms matched in Dictionary ({len(expanded_terms)}):**")
            st.code(expanded_terms)

            st.markdown(f"**Matching Documents Retrieved ({len(wildcard_docs)}):**")
            if wildcard_docs:
                for doc in wildcard_docs:
                    st.success(f"✅ {engine.doc_names[doc]} ({doc})")
            else:
                st.warning("No documents contain terms matching this wildcard pattern.")

elif tolerant_mode == "Spelling Correction (Edit Distance)":
    st.markdown("### ✏️ Spell-Correction Engine")
    st.write("When a query term is misspelled (not found in the dictionary), the system calculates the Levenshtein Edit Distance against all terms in the vocabulary and suggests closest matches. This is the underlying algorithm for modern 'Did you mean?' suggestions.")

    typo_in = st.text_input("Type a word with a typo (e.g. 'inforamtion' or 'retreival')", "inforamtion")

    if typo_in:
        typo_clean = typo_in.lower().strip()

        if typo_clean in engine.vocab_raw:
            st.success(f"🎯 The word '{typo_clean}' is in the dictionary! No correction needed.")
        else:
            st.warning(f"⚠️ '{typo_clean}' is NOT in the vocabulary.")

            suggestions = engine.get_spelling_suggestions(typo_clean, max_dist=2)

            if suggestions:
                st.markdown("**Did you mean?**")
                cols = st.columns(len(suggestions))
                for idx, sug in enumerate(suggestions):
                    with cols[idx]:
                        if st.button(f"👉 {sug}", key=f"sug_btn_{sug}"):
                            lemmed = engine.lemmatizer.lemmatize(sug) if engine.lemmatizer else sug
                            matching_docs = engine.inverted_index_lem.get(lemmed, [])

                            st.success(f"Searching for corrected term: '{sug}'")
                            if matching_docs:
                                st.write("Matching documents:")
                                for doc in matching_docs:
                                    st.write(f"- {engine.doc_names[doc]} ({doc})")
                            else:
                                st.write("No matching documents.")

                st.markdown("#### Edit Distance Computation Table")
                matrix_data = []
                for sug in suggestions:
                    matrix_data.append({
                        "Typed Input": typo_clean,
                        "Dictionary Term Candidate": sug,
                        "Levenshtein Edit Distance": engine.edit_distance(typo_clean, sug)
                    })
                st.table(matrix_data)

            else:
                st.error("No vocabulary terms are close enough (Edit Distance > 2) to suggest a correction.")

elif tolerant_mode == "Phonetic Correction (Soundex)":
    st.markdown("### \U0001f50a Phonetic Query Correction (Soundex Algorithm)")
    st.write("The Soundex phonetic algorithm encodes words by how they sound rather than spelling. Words that sound alike receive the same Soundex code, enabling retrieval of phonetically similar vocabulary terms.")

    col_algo, col_demo = st.columns([1, 1])

    with col_algo:
        st.markdown("#### How Soundex Works")
        st.markdown("""
        Algorithm (American Soundex Standard):
        1. Retain the first letter (uppercased).
        2. Replace remaining consonants with digits:
           - B, F, P, V -> 1 | C, G, J, K, Q, S, X, Z -> 2
           - D, T -> 3 | L -> 4 | M, N -> 5 | R -> 6
        3. Remove vowels (A/E/I/O/U) and H, W, Y after first letter.
        4. Collapse adjacent identical codes into one digit.
        5. Pad with zeros or truncate to 4 characters (letter + 3 digits).

        Example: information and inforamtion (typo) both produce I516!
        Example: retrieval and retreival (typo) both produce R361!
        """)

    with col_demo:
        st.markdown("#### Live Phonetic Lookup")
        phonetic_in = st.text_input("Enter a word (correct or misspelled) to find phonetic matches:", "inforamtion", key="phonetic_input")

        if phonetic_in:
            phonetic_clean = phonetic_in.lower().strip()
            soundex_code = engine.soundex(phonetic_clean)
            st.info(f"**Soundex Code** for '{phonetic_clean}': *{soundex_code}*")

            matches, query_code = engine.get_phonetic_suggestions(phonetic_clean)

            if phonetic_clean in engine.vocab_raw:
                st.success(f"The term '{phonetic_clean}' is already in the vocabulary.")
            else:
                st.warning(f"'{phonetic_clean}' is NOT in the vocabulary.")

            if matches:
                st.markdown(f"**Phonetically Similar Terms (Soundex Code = {soundex_code}):**")
                for m in matches:
                    m_code = engine.soundex(m)
                    st.write(f"- {m} (code: {m_code})")

                st.markdown("**Documents containing phonetically matched terms:**")
                all_docs = set()
                for m in matches:
                    lemmed = engine.lemmatizer.lemmatize(m) if engine.lemmatizer else m
                    docs = engine.inverted_index_lem.get(lemmed, [])
                    all_docs.update(docs)

                if all_docs:
                    for doc in sorted(all_docs):
                        st.success(f"Document: {engine.doc_names[doc]} ({doc})")
                else:
                    st.warning("No documents found for phonetic matches.")
            else:
                st.error(f"No vocabulary terms share Soundex code '{soundex_code}' with '{phonetic_clean}'.")

    st.divider()
    st.subheader("Phonetic Index Explorer")
    st.write("Shows vocabulary groupings by Soundex code to demonstrate phonetic clustering:")

    import pandas as pd
    phonetic_index = engine.get_phonetic_index()
    interesting = {c: t for c, t in phonetic_index.items() if len(t) > 1}
    interesting = dict(sorted(interesting.items(), key=lambda x: -len(x[1]))[:15])

    if interesting:
        rows = [{"Soundex Code": c, "Phonetically Similar Terms": ", ".join(t), "Group Size": len(t)} for c, t in interesting.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("**Inference:** Soundex groups phonetically similar words under one code. This allows the IR system to expand queries and retrieve documents even when users use phonetic spelling or mispronounce technical terms.")
    else:
        st.write("No phonetic groups with multiple terms found in current corpus.")

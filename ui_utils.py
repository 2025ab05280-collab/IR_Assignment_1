import streamlit as st

def render_page_header(title, subtitle, icon="🔍"):
    # Using pure native Streamlit elements
    st.title(f"{icon} {title}")
    st.caption(subtitle)
    st.divider()

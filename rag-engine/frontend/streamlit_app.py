"""
Query dashboard (Phase 5, step 2).
Lets the user ask a question and see:
  - the generated answer with clickable citations
  - retrieved chunks ranked by relevance
  - confidence scores broken down by dimension
  - a toggle to compare hybrid vs. dense-only retrieval side by side
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Engine", layout="wide")
st.title("RAG Engine — Query Dashboard")

question = st.text_input("Ask a question about the indexed documents")
mode = st.radio("Retrieval mode", ["hybrid", "dense_only"], horizontal=True)
compare = st.checkbox("Compare hybrid vs. dense-only side by side")

if st.button("Ask") and question:
    # TODO: POST to {API_BASE_URL}/v1/ask, render:
    #   - answer with citation markers
    #   - retrieved chunks table (ranked, with relevance scores)
    #   - confidence breakdown (retrieval / citation coverage / completeness)
    #   - if `compare`, run both modes and show two columns side by side
    st.info("Wire this up to POST /v1/ask once the API is implemented.")

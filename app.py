import streamlit as st
import tempfile
from pathlib import Path
from chunk_and_index import generate_data
from langchain_community.document_loaders import TextLoader
from query_data import main

st.set_page_config(page_title="Financial Report RAG")
st.title("Financial Report RAG")

uploaded_file = st.file_uploader("Upload a `.txt` financial report", type=["txt"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = Path(tmp_file.name)

    st.success("File uploaded successfully")

    if st.button("Run Chunk + Index"):
        with st.spinner("Processing document..."):
            loader = TextLoader(str(tmp_path), encoding="utf-8")
            docs = loader.load()
            generate_data(docs)
            st.success("Chunking and indexing complete")
query_text = st.text_input("Enter your query")
if st.button("Query Data") and query_text:
    with st.spinner("Processing query..."):
        response = main(query_text)
        st.success("Query complete")
        st.write(response)
        
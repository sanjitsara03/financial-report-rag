import streamlit as st
import tempfile
from pathlib import Path
from chunk_and_index import generate_data
from langchain_community.document_loaders import TextLoader
from query_data import main
from pdf_to_img import pdf_to_images
from img_to_md import process_images_to_markdown
from combine_md import stitch_markdown_pages

st.set_page_config(page_title="Financial Report RAG")

st.title("Financial Report RAG")

# uploaded_file = st.file_uploader("Upload a `.txt` financial report", type=["txt"])
tab1, tab2 = st.tabs(["Text File", "PDF File"])

with tab1:
    uploaded_txt_file = st.file_uploader("Upload a `.txt` financial report", type=["txt"])
    if uploaded_txt_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_txt_file.read())
            tmp_path = Path(tmp_file.name)

        st.success("File uploaded successfully")

        

with tab2:
    uploaded_pdf_file = st.file_uploader("Upload a `.pdf` financial report", type=["pdf"])
    if uploaded_pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf_file.read())
            tmp_path = Path(tmp_file.name)
        st.success("File uploaded successfully")
        if st.button("Convert PDF to MD"):
            with st.spinner("Processing document..."):
                pdf_to_images(str(tmp_path))
                process_images_to_markdown()
                stitch_markdown_pages('page_markdowns', 'combined_txt.txt')
                st.success("PDF to MD complete")
    
            

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
        
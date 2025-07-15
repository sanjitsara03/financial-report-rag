
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import re, json
import openai 
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

TABLE_RE = re.compile(
    r'(\|[^\n]*\|\n'           
    r'(?:\|[^\n]*\|\n?)*)',    
    re.MULTILINE
)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
def generate_data(docs: list[Document]):
    table_chunks, text_chunks = split_text(docs)
    create_db(table_chunks, text_chunks)

def split_text(docs: list[Document]):
   
    text_chunks: list[Document] = []
    table_chunks: list[Document] = []
    text_id = 1  
    table_id = 1  
    table_context_id = 1  
    for doc in docs:
        text = doc.page_content
        meta = doc.metadata.copy()                 
        last_end = 0                               

        for m in TABLE_RE.finditer(text):
            text_block = text[last_end:m.start()]
            for chunk in SPLITTER.split_text(text_block):
                text_chunks.append(
                    Document(page_content=chunk,
                             metadata={**meta, "text_id": text_id})
                )
                text_id += 1
            
            text_context_id = text_id if text_block.strip() else -1
            if text_context_id == -1:
                table_context_id = table_id

            table = m.group(0).strip()
            table_id += 1
            table_chunks.append(
                Document(page_content=table,
                         metadata={**meta,"table_id": table_id, "text_context_id": text_context_id, "table_context_id": table_context_id})
            )
            
            last_end = m.end()

        trailing = text[last_end:]
        for chunk in SPLITTER.split_text(trailing):
            text_chunks.append(
                Document(page_content=chunk,
                         metadata={**meta, "text_id": text_id})
            )
            text_id += 1
      
    return text_chunks, table_chunks

def create_db(text_chunks, table_chunks):
    embeddings = OpenAIEmbeddings()
    persist_directory = "text_db"
    db = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)
    persist_directory = "table_db"
    db = Chroma.from_documents(table_chunks, embeddings, persist_directory=persist_directory)
    

if __name__ == "__main__":
    FILE_PATH = Path("combined_txt.txt")   

    loader = TextLoader(str(FILE_PATH), encoding="utf-8")
    original_docs = loader.load()
    generate_data(original_docs)
 
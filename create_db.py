
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
    r'\|[^\n]*[-|][^\n]*\|\n'  
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
    create_db(text_chunks, table_chunks)

def split_text(docs: list[Document], json_out: str = "table_index.json"):
   
    text_chunks: list[Document] = []
    table_chunks: list[Document] = []
    table_links: list[dict] = []
    chunk_id = 0  

    for doc in docs:
        text = doc.page_content
        meta = doc.metadata.copy()                 
        last_end = 0                               

        for m in TABLE_RE.finditer(text):
            text_block = text[last_end:m.start()]
            for chunk in SPLITTER.split_text(text_block):
                chunk_id += 1
                text_chunks.append(
                    Document(page_content=chunk,
                             metadata={**meta, "chunk_id": chunk_id})
                )

            context_id = chunk_id if text_block.strip() else (context_id if context_id else None)

            table = m.group(0).strip()
            table_chunks.append(
                Document(page_content=table,
                         metadata={**meta, "text_chunk_id": context_id})
            )
            table_links.append({
                "table": table,
                "context_chunk_id": context_id
            })
            last_end = m.end()

        trailing = text[last_end:]
        for chunk in SPLITTER.split_text(trailing):
            chunk_id += 1
            text_chunks.append(
                Document(page_content=chunk,
                         metadata={**meta, "chunk_id": chunk_id})
            )

    #Write table links to json
    Path(json_out).write_text(
        json.dumps(table_links, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    #Debugging
    for chunk in table_chunks:
        print(chunk.page_content)
        print(chunk.metadata)
        print("--------------------------------")
    for chunk in text_chunks:
        print(chunk.page_content)
        print(chunk.metadata)
        print("--------------------------------")
    return text_chunks, table_chunks

def create_db(text_chunks, table_chunks):
    persist_directory = "text_db"
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)
    persist_directory = "table_db"
    db = Chroma.from_documents(table_chunks, embeddings, persist_directory=persist_directory)
    

if __name__ == "__main__":
    FILE_PATH = Path("combined_txt_test.txt")   

    loader = TextLoader(str(FILE_PATH), encoding="utf-8")
    original_docs = loader.load()
    generate_data(original_docs)
 
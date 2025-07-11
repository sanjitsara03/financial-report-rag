import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH_TABLE = "table_db"
CHROMA_PATH_TEXT = "text_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context: 

{context}

---

Question:  {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings()
    db_table = Chroma(persist_directory=CHROMA_PATH_TABLE, embedding_function=embedding_function)
    db_text = Chroma(persist_directory=CHROMA_PATH_TEXT, embedding_function=embedding_function)

    results = db_table.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(results[0][1])
        print(f"Unable to find matching results.")
        return
    context_text = ""
    for table_doc,score in results:
        text_id = table_doc.metadata["text_chunk_id"]
        text_results = db_text.similarity_search("placeholder", k=1, filter={"chunk_id": text_id})
        text_doc = text_results[0] if text_results else None
        if text_doc:
            context_text += text_doc.page_content + "\n"
            context_text += table_doc.page_content + "\n\n---\n\n"
    

    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke(prompt)
    response_text = response.content

    response = f"Response: {response_text}\n"
    print(response)


if __name__ == "__main__":
    main()
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


def main(query_text=None):
    if query_text is None:
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
        context_text = table_doc.page_content + "\n\n---\n\n" + context_text
        text_id = table_doc.metadata["text_context_id"]
        table_doc_dup = table_doc

        if int(text_id) == -1:
            
            while(int(text_id) == -1):
         
                table_id = table_doc_dup.metadata["table_context_id"]
                table_result = db_table.similarity_search("placeholder", k=1, filter={"table_id": table_id})
                table_doc_2 = table_result[0] if table_result else None
                if not table_doc_2:
                    print("No table found")
                if table_doc_2:
                    context_text = table_doc_2.page_content + "\n" + context_text
                    text_id = table_doc_2.metadata["text_context_id"]                   
                    table_doc_dup = table_doc_2
        text_results = db_text.similarity_search("placeholder", k=1, filter={"text_id": text_id})
        text_doc = text_results[0] if text_results else None
        if text_doc:
            context_text = text_doc.page_content + "\n" + context_text
            
    

    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke(prompt)
    response_text = response.content

    response = f"Response: {response_text}\n"
    print(response)
    return response


if __name__ == "__main__":
    main()
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

# Load the FAISS Index
INDEX_PATH = "asyncapi_faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
docsearch = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Define the LLM
chat = ChatOpenAI(model="gpt-4", temperature=0)

prompt_template = """You are an expert in AsyncAPI and event-driven architectures. 
Use the provided context to answer the user's query. Always provide complete examples when applicable.

Context:
-------------------
{context}

User Query: {input}

Response Guidelines:
1. For specification questions: Provide complete YAML examples with all required fields
2. For implementation questions: Provide JavaScript code examples
3. For conceptual questions: Explain clearly and reference diagrams when available
4. Always format code properly:

Response format:
[Explanation] A brief explanation of the concept.
[Code Example] (if applicable):
```yaml
#code here
```
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

stuff_document_chain = create_stuff_documents_chain(chat, prompt)


history_aware_retriever = create_history_aware_retriever(
    llm=chat,
    retriever=docsearch.as_retriever(),
    prompt=prompt
)

qa_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=stuff_document_chain
)

def query_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Queries the LLM assistant with context-aware retrieval."""
    result = qa_chain.invoke({"input": query, "chat_history": chat_history})
    return {
    "answer": result["answer"], # Changed from "result" to "answer"
    "query": query,
    "sources": [doc.metadata for doc in result.get("context", [])]
}

if __name__ == "__main__":
    res = query_llm(query="Explain AsyncAPI channels with examples")
    print(res["answer"])
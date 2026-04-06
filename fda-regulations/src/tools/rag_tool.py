import os
from typing import List
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool

embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db_path = os.getenv("VECTOR_DB_PATH", "/app/data/vector_db")
vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)

# @tool - For now keep as a simple function call, but this can be expanded to a full LangChain tool with structured outputs if needed.
def search_fda_precedents(query: str, k: int = 3) -> List[str]:
    """
    Searches the FDA Warning Letter database for atomic deficiencies
    matching the provided query. Returns the top 3 most relevant matches
    including Regulation, Violation, Description, and Evidence.

    Args:
        query (str): A plain text search string describing compliance concerns or regulatory topics,
                     e.g., '21 CFR 211 cleaning validation sterile manufacturing'
        k (int, optional): Number of top matches to return. Defaults to 3.

    Returns:
        List[str]: List of up to k matching deficiency descriptions as formatted strings,
                   each containing CFR reference, violation title, description, evidence,
                   and required action from historical FDA warning letters
    """
    results = vector_db.similarity_search(query, k)
    return [r.page_content for r in results]
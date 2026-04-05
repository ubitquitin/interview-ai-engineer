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
        query: A plain text search string, e.g.
               '21 CFR 211 cleaning validation sterile manufacturing'

    Returns:
        List of up to 3 matching deficiency descriptions as strings.
    """
    results = vector_db.similarity_search(query, k)
    return [r.page_content for r in results]
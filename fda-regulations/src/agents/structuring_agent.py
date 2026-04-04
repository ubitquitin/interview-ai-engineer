import os
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from src.models import InputRequest, ExtractionOutput

app = FastAPI()
# Use OLLAMA_BASE_URL environment variable if available
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="llama3", temperature=0, format="json", base_url=ollama_base_url)

@app.post("/agent/structuring")
async def run_structuring_agent(request: InputRequest):
    # Access the text via request.input_text
    structured_llm = llm.with_structured_output(ExtractionOutput)

    prompt = f"""
    You are a GMP Systems Architect. Deconstruct the following regulatory process description
    into a structured profile. Extract all entities, systems, and controls.

    INPUT TEXT:
    {request.input_text}
    """

    result = await structured_llm.ainvoke(prompt)
    return result

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "structuring-agent"}
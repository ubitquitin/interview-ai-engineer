import os
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.tools.rag_tool import search_fda_precedents
from src.models import ComplianceOutput, ExtractionOutput

app = FastAPI()

# 1. Setup Persona and Instructions
system_prompt = """
You are a Senior FDA Compliance Auditor. Your goal is to identify regulatory risks
in pharmaceutical processes by comparing user data against historical 2026 Warning Letters.

PROCEDURE:
1. Review the structured process provided by the user.
2. Use the 'search_fda_precedents' tool to find specific 21 CFR violations.
3. Compare the user's 'controls' against the 'evidence' in the Warning Letters.
4. Provide a final assessment.

You MUST call the search tool at least once to ground your analysis in 2026 data.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 2. Initialize LLM and Agent
# Note: Llama 3 8B via Ollama supports tool calling in recent versions.
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="llama3", temperature=0, base_url=ollama_base_url)
tools = [search_fda_precedents]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/agent/compliance")
async def run_compliance_agent(structured_input: ExtractionOutput):
    # Pass 1: Agentic Reasoning & Tool Usage
    # The agent will search ChromaDB and summarize findings.
    agent_input = f"Perform a compliance audit for this process: {structured_input.json()}"
    raw_result = await agent_executor.ainvoke({"input": agent_input})

    # Pass 2: Structured Output Enforcement
    # Agents are messy; we use a secondary 'Structuring' call to ensure
    # the output perfectly matches your ComplianceOutput Pydantic model.
    structured_llm = llm.with_structured_output(ComplianceOutput)

    final_report = await structured_llm.ainvoke(
        f"Based on this research: {raw_result['output']}, "
        f"format the audit into the required JSON schema for the user's process: {structured_input.json()}"
    )

    return final_report

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "compliance-agent"}
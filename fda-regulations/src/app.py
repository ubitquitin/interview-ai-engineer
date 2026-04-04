"""
Unified FastAPI application that runs all agents and the pipeline orchestrator.
This simplifies deployment by having a single service instead of multiple microservices.
"""
import os
import httpx
from fastapi import FastAPI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.models import ExtractionOutput, ComplianceOutput, InputRequest
from src.tools.rag_tool import search_fda_precedents

# Initialize FastAPI app
app = FastAPI(
    title="FDA Compliance Pipeline",
    description="Unified service for structuring and compliance agents",
    version="1.0.0"
)

# Ollama configuration - defaults to host machine when running in Docker
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_BASE_URL)

# ============================================================================
# STRUCTURING AGENT
# ============================================================================

@app.post("/agent/structuring", response_model=ExtractionOutput)
async def run_structuring_agent(request: InputRequest):
    """
    Structuring Agent - extracts entities, systems, and controls from raw text.
    """
    structured_llm = llm.with_structured_output(ExtractionOutput)

    prompt = f"""
    You are a GMP Systems Architect. Deconstruct the following regulatory process description
    into a structured profile. Extract all entities, systems, and controls.

    INPUT TEXT:
    {request.input_text}
    """

    result = await structured_llm.ainvoke(prompt)
    return result


# ============================================================================
# COMPLIANCE AGENT
# ============================================================================

# Setup compliance agent prompt
compliance_system_prompt = """
You are a Senior FDA Compliance Auditor. Your goal is to identify regulatory risks
in pharmaceutical processes by comparing user data against historical 2026 Warning Letters.

PROCEDURE:
1. Review the structured process provided by the user.
2. Use the 'search_fda_precedents' tool to find specific 21 CFR violations.
3. Compare the user's 'controls' against the 'evidence' in the Warning Letters.
4. Provide a final assessment.

You MUST call the search tool at least once to ground your analysis in 2026 data.
"""

compliance_prompt = ChatPromptTemplate.from_messages([
    ("system", compliance_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Initialize compliance agent
compliance_tools = [search_fda_precedents]
compliance_agent = create_tool_calling_agent(llm, compliance_tools, compliance_prompt)
agent_executor = AgentExecutor(agent=compliance_agent, tools=compliance_tools, verbose=True)


@app.post("/agent/compliance", response_model=ComplianceOutput)
async def run_compliance_agent(structured_input: ExtractionOutput):
    """
    Compliance Agent - performs RAG-based risk assessment against FDA precedents.
    """
    # Pass 1: Agentic Reasoning & Tool Usage
    agent_input = f"Perform a compliance audit for this process: {structured_input.json()}"
    raw_result = await agent_executor.ainvoke({"input": agent_input})

    # Pass 2: Structured Output Enforcement
    structured_llm = llm.with_structured_output(ComplianceOutput)
    final_report = await structured_llm.ainvoke(
        f"Based on this research: {raw_result['output']}, "
        f"format the audit into the required JSON schema for the user's process: {structured_input.json()}"
    )

    return final_report


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class PipelineState(TypedDict):
    input: str
    structured: dict
    compliance: dict


async def structuring_node(state: PipelineState):
    """Call the structuring agent endpoint internally"""
    request = InputRequest(input_text=state["input"])
    result = await run_structuring_agent(request)
    return {"structured": result.dict()}


async def compliance_node(state: PipelineState):
    """Call the compliance agent endpoint internally"""
    structured_input = ExtractionOutput(**state["structured"])
    result = await run_compliance_agent(structured_input)
    return {"compliance": result.dict()}


def build_graph():
    workflow = StateGraph(PipelineState)
    workflow.add_node("structuring", structuring_node)
    workflow.add_node("compliance", compliance_node)

    workflow.set_entry_point("structuring")
    workflow.add_edge("structuring", "compliance")
    workflow.add_edge("compliance", END)

    return workflow.compile()


graph = build_graph()


@app.post("/pipeline")
async def full_pipeline(request: InputRequest):
    """
    Full pipeline - orchestrates structuring and compliance agents using LangGraph.
    """
    inputs = {"input": request.input_text}
    result = await graph.ainvoke(inputs)
    return result


# ============================================================================
# HEALTH CHECKS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fda-compliance-pipeline",
        "ollama_url": OLLAMA_BASE_URL
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "FDA Compliance Pipeline",
        "version": "1.0.0",
        "endpoints": {
            "pipeline": "/pipeline",
            "structuring_agent": "/agent/structuring",
            "compliance_agent": "/agent/compliance",
            "health": "/health",
            "docs": "/docs"
        }
    }

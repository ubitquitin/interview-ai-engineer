"""
Unified FastAPI application that runs all agents and the pipeline orchestrator.
Single-service, multi-agent pipeline using LangGraph + tool binding (no AgentExecutor).
"""
import os
from fastapi import FastAPI
from langgraph.graph import StateGraph, END
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.models import ExtractionOutput, ComplianceOutput, InputRequest
from src.tools.rag_tool import search_fda_precedents

# ----------------------------------------------------------------------------
# APP INIT
# ----------------------------------------------------------------------------

app = FastAPI(
    title="FDA Compliance Pipeline",
    description="Unified service for structuring and compliance agents",
    version="1.0.0"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url=OLLAMA_BASE_URL
)

# ----------------------------------------------------------------------------
# STRUCTURING AGENT
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# COMPLIANCE AGENT (TOOL BINDING VERSION)
# ----------------------------------------------------------------------------

compliance_system_prompt = """
You are a Senior FDA Compliance Auditor.

PROCEDURE:
1. Review the structured process.
2. You MUST call the 'search_fda_precedents' tool EXACTLY ONCE.
3. Use retrieved FDA Warning Letter evidence.
4. Assess compliance risks.

Do not skip the tool call.
"""

compliance_prompt = ChatPromptTemplate.from_messages([
    ("system", compliance_system_prompt),
    ("human", "{input}")
])

compliance_tools = [search_fda_precedents]
llm_with_tools = llm.bind_tools(compliance_tools)


@app.post("/agent/compliance", response_model=ComplianceOutput)
async def run_compliance_agent(structured_input: ExtractionOutput):
    """
    Compliance Agent - performs RAG-based risk assessment against FDA precedents.
    """

    # ------------------------------------------------------------------------
    # PASS 1: TOOL CALLING
    # ------------------------------------------------------------------------

    agent_input = f"Perform a compliance audit:\n{structured_input.json()}"

    response = await llm_with_tools.ainvoke(
        compliance_prompt.format_messages(input=agent_input)
    )

    tool_outputs = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_fda_precedents":
                args = tool_call["args"]
                # Coerce query to string if model passed a dict
                if isinstance(args.get("query"), dict):
                    args["query"] = " ".join(
                        f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                        for k, v in args["query"].items()
                    )
                
                result = search_fda_precedents.invoke(args)
                tool_outputs.append(str(result))

    if not tool_outputs:
        raise ValueError("Model failed to call required tool")

    # ------------------------------------------------------------------------
    # PASS 2: STRUCTURED OUTPUT
    # ------------------------------------------------------------------------

    structured_llm = llm.with_structured_output(ComplianceOutput)

    final_report = await structured_llm.ainvoke(f"""
Structured process:
{structured_input.json()}

FDA precedents:
{tool_outputs}

Return a full compliance assessment in the required schema.
""")

    return final_report


# ----------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR (LANGGRAPH)
# ----------------------------------------------------------------------------

class PipelineState(TypedDict):
    input: str
    structured: dict
    compliance: dict


async def structuring_node(state: PipelineState):
    request = InputRequest(input_text=state["input"])
    result = await run_structuring_agent(request)
    return {"structured": result.dict()}


async def compliance_node(state: PipelineState):
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
    Full pipeline - structuring → compliance
    """
    inputs = {"input": request.input_text}
    result = await graph.ainvoke(inputs)
    return result


# ----------------------------------------------------------------------------
# HEALTH + ROOT
# ----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "fda-compliance-pipeline",
        "ollama_url": OLLAMA_BASE_URL
    }


@app.get("/")
async def root():
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
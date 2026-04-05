"""
Unified FastAPI application that runs all agents and the pipeline orchestrator.
Single-service, multi-agent pipeline using LangGraph + tool binding.
"""
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Type
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.runnables import RunnableConfig

from src.models import ExtractionOutput, ComplianceOutput, InputRequest
from src.tools.rag_tool import search_fda_precedents

# ----------------------------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FDA-Pipeline")

app = FastAPI(
    title="FDA Compliance Pipeline",
    description="Unified service for structuring and compliance agents",
    version="1.0.0"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# The LLM Base for agents. Replace with your actual model and config as needed.
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url=OLLAMA_BASE_URL,
    num_thread=8
)

# Using ConsoleCallbackHandler to log all LLM interactions to the console for debugging and observability.
config = RunnableConfig(callbacks=[ConsoleCallbackHandler()])

# ----------------------------------------------------------------------------
# AGENT HELPERS (RETRY LOGIC)
# ----------------------------------------------------------------------------
def get_structured_llm_with_retry(schema: Type[BaseModel]) -> Any:
    """
    Creates a structured LLM with a retry policy for schema validation errors.

    Args:
        schema: Pydantic model class for structured output validation.

    Returns:
        Configured LLM chain with retry logic and structured output parsing.
    """
    return llm.with_structured_output(schema).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )

# ----------------------------------------------------------------------------
# AGENT LOGIC
# ----------------------------------------------------------------------------
@app.post("/agent/structuring", response_model=ExtractionOutput)
async def run_structuring_agent(request: InputRequest):
    '''
    Agent that takes raw input text and extracts structured information about entities, systems, controls, processes, and materials relevant to GMP compliance.
    Implements retry logic for schema validation errors to enhance robustness.
    '''
    start_time = time.time()
    logger.info("Starting Structuring Agent...")
    
    structured_llm = get_structured_llm_with_retry(ExtractionOutput)

    prompt = f"You are a GMP Systems Architect. Extract entities, systems, and controls:\n{request.input_text}"

    try:
        # Using config to enable ConsoleCallbackHandler
        result = await structured_llm.ainvoke(prompt, config=config)
        logger.info(f"Structuring Agent finished in {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Structuring Agent failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Structuring failed after retries")


@app.post("/agent/compliance", response_model=ComplianceOutput)
async def run_compliance_agent(structured_input: ExtractionOutput):
    '''
    Agent that takes structured information about a GMP process and assesses compliance risks and potential FDA citations.
    Integrates a RAG tool to retrieve relevant FDA precedents based on the structured input, and uses this information to inform the compliance assessment.
    '''
    start_time = time.time()
    logger.info("Starting Compliance Agent...")
    
    query_str = f'''
    Summary: {structured_input.summary}

    Entities: {[f"{entity.name}: {entity.description}" for entity in structured_input.entities]}

    Systems: {structured_input.systems if structured_input.systems else "None"}

    Controls: {structured_input.controls if structured_input.controls else "None"}

    Processes: {structured_input.processes if structured_input.processes else "None"}

    Materials: {structured_input.materials if structured_input.materials else "None"}
    '''

    tool_outputs = search_fda_precedents(query_str)
    logger.log(logging.DEBUG, f"RAG Tool Output: {tool_outputs}")
    # TODO: Bind the actual tool to the model and parse tool calls instead of invoking directly here.
    # llm_with_tools = llm.bind_tools([search_fda_precedents])

    structured_llm = get_structured_llm_with_retry(ComplianceOutput)
    
    final_report = await structured_llm.ainvoke(f"""
        You are a GMP Compliance Officer. Assess the following structured process description for compliance risks and potential FDA citations. 
        Use the provided FDA precedents to inform your analysis, but focus on the specific entities, systems, controls, processes, and materials described in the input.
        FDA precedents: {tool_outputs}
        Input: {structured_input.json()}
        Return the final assessment.
    """, config=config)

    logger.info(f"Compliance Agent finished in {time.time() - start_time:.2f}s")
    return final_report

# ----------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ----------------------------------------------------------------------------
class PipelineState(TypedDict):
    '''
    This class keeps track of data flowing through the Agentic pipeline.
    Includes the input, the structured output from the structuring agent, 
    and the compliance assessment from the compliance agent.
    '''
    input: str
    structured: dict[str, Any]  # ExtractionOutput serialized to dict
    compliance: dict[str, Any]  # ComplianceOutput serialized to dict


async def structuring_node(state: PipelineState) -> dict[str, dict[str, Any]]:
    '''
    Node that runs the structuring agent and updates the pipeline state with the structured output.
    '''
    request = InputRequest(input_text=state["input"])
    result = await run_structuring_agent(request)
    return {"structured": result.dict()}


async def compliance_node(state: PipelineState) -> dict[str, dict[str, Any]]:
    '''
    Node that runs the compliance agent and updates the pipeline state with the compliance assessment.
    '''
    structured_input = ExtractionOutput(**state["structured"])
    result = await run_compliance_agent(structured_input)
    return {"compliance": result.dict()}


def build_graph() -> StateGraph[PipelineState]:
    """
    Constructs the LangGraph pipeline workflow.

    Returns:
        Compiled graph with structuring -> compliance nodes.
    """
    workflow = StateGraph(PipelineState)
    workflow.add_node("structuring", structuring_node)
    workflow.add_node("compliance", compliance_node)
    workflow.set_entry_point("structuring")
    workflow.add_edge("structuring", "compliance")
    workflow.add_edge("compliance", END)
    return workflow.compile()

# Builds the langgraph once at startup to avoid overhead on each request. 
# The graph is stateless and can be reused across requests.
graph = build_graph()

@app.post("/pipeline")
async def full_pipeline(request: InputRequest):
    logger.info("--- PIPELINE START ---")
    inputs = {"input": request.input_text}
    result = await graph.ainvoke(inputs)
    logger.info("--- PIPELINE END ---")
    return result

# ----------------------------------------------------------------------------
# HEALTH + ROOT
# ----------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "fda-compliance-pipeline",
        "ollama_url": OLLAMA_BASE_URL
    }


@app.get("/")
async def root() -> dict[str, str | dict[str, str]]:
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
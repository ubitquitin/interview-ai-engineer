from datetime import date
from enum import Enum
from typing import List, Optional, Literal, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, HttpUrl
import re

'''
Parsing Data Structures
'''
class Deficiency(BaseModel):
    """
    A specific regulatory failure cited by the FDA.
    This is the core unit of comparison for the compliance agent. 
    A given letter may contain many Defiencies.
    """
    title: str = Field(description="The bolded header/violation name")
    cfr_reference: str = Field(description="The cfr reference number like '21 CFR 211.68'")
    description: str = Field(description="The factual narrative of the failure")
    evidence: str = Field(description="Specific observations or data points mentioned")
    required_action: str = Field(description="The remediation the FDA demands")


class WarningLetterMetadata(BaseModel):
    """Represents an Index/metadata for an FDA Warning Letter."""
    company_name: str
    issue_date: str
    url: HttpUrl
    issuing_office: str
    subject: Optional[str] = None
    ref_number: Optional[str] = None


class WarningLetterDocument(BaseModel):
    """Represents an FDA warning letter with structured metadata and content."""
    metadata: WarningLetterMetadata
    introduction: str
    deficiencies: List[Deficiency]
    conclusion: str
    content_hash: str


'''
Agent Data Structures
'''
class InputRequest(BaseModel):
    '''
    Input request for the structuring agent/pipeline. 
    Contains the raw text to be processed.
    '''
    input_text: str
    
    #TODO: Add in guardrails as functions to filter out malicious input text.


class ProcessEntity(BaseModel):
    """
    A normalized unit of context extracted from raw input.
    Entities should reflect components an FDA inspector would evaluate.
    """
    name: str = Field(
        description="Canonical name of the entity (e.g., 'Batch Record System', 'Weighing Process')"
    )

    type: str = Field(
        description="Entity category. One of: process, system, material, control."
    )

    description: str = Field(
        description="Concise explanation of the entity and its role in the workflow."
    )


class ExtractionOutput(BaseModel):
    """
    Structured representation of a pharma or regulated workflow.

    This output is designed to normalize messy natural language into
    inspection-relevant components for downstream compliance analysis.
    """
    summary: str = Field(
        description="High-level summary of the overall process or system described in the input."
    )

    entities: List[ProcessEntity] = Field(
        description="All extracted entities, normalized into structured objects."
    )

    processes: List[str] = Field(
        description="List of operational processes (e.g., mixing, sterilization, batch release)."
    )

    materials: List[str] = Field(
        description="List of materials, substances, or inputs used in the process."
    )

    systems: List[str] = Field(
        description="List of systems (software, equipment, infrastructure) involved."
    )

    controls: List[str] = Field(
        description="List of quality or compliance controls (e.g., audit trails, validation steps, SOPs)."
    )


class RiskLevel(str, Enum):
    """
    Standardized FDA-style risk classification.

    HIGH:
        Likely to trigger regulatory action or Form 483 observation.
        Typically indicates missing controls, data integrity risks, or GMP violations.

    MEDIUM:
        Some controls exist but gaps or weaknesses are present.
        Could escalate under inspection scrutiny.

    LOW:
        Minor issues or unlikely edge cases.
        Generally compliant but not robust.

    NONE:
        No meaningful risk identified.
        Fully aligned with expected regulatory standards.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


"""
Outputs from Compliance Agent. Risk Item wraps an identified potential deficiency with risk level and agent reasoning.
Compliance output is the overall list of RiskItems with an overall summary and overall risk score.
"""
class RiskItem(BaseModel):
    """
    A single compliance risk derived from comparison with historical FDA deficiencies.

    Each item represents a potential regulatory issue mapped to a known CFR violation.
    """
    deficiency: "Deficiency" = Field(
        description="The historical FDA deficiency this risk is based on."
    )

    risk_level: RiskLevel = Field(
        description="Severity of the risk based on similarity to known violations."
    )

    reasoning: str = Field(
        description=(
            "Detailed explanation of why this risk level was assigned. "
            "Should explicitly compare the input process to the deficiency and highlight gaps."
        )
    )


class ComplianceOutput(BaseModel):
    """
    Final compliance assessment of the provided process.

    Aggregates all identified risks and provides an overall regulatory risk posture.
    """
    risks: List[RiskItem] = Field(
        description="List of all identified risks mapped to FDA deficiencies."
    )

    overall_risk: RiskLevel = Field(
        description="Overall risk level considering all identified issues."
    )

    summary: str = Field(
        description=(
            "Executive-level summary of compliance posture, key risks, and major concerns. "
            "Should be readable by a regulatory or quality leader."
        )
    )

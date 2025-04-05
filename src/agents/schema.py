
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# Define output schema
class EligibilityAgentOutput(BaseModel):
    eligible: bool = Field(description="Whether the company is eligible for this RFP")
    reasons: List[str] = Field(description="List of reasons for eligibility decision")
    missing_requirements: Optional[List[str]] = Field(
        default=None, 
        description="Requirements that the company does not meet"
    )
    matching_requirements: Optional[List[str]] = Field(
        default=None, 
        description="Requirements that the company successfully meets"
    )

class AgentAction(BaseModel):
    """Model for tracking agent actions and decisions"""
    action: str = Field(description="Type of action taken by the agent")
    reasoning: str = Field(description="Reasoning behind the action")
    query: str = Field(description="Query used for retrieval or analysis")
    result: Any = Field(description="Result of the action")


class ComplianceChecklistOutput(BaseModel):
    checklist: dict = Field(description="Dictionory which contains extracted compliance checklist")
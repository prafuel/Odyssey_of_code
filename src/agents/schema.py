
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
    recommended_actions: List = Field(
        default=None,
        description="Recommended actions to improve eligibility"
    )

class AgentAction(BaseModel):
    """Model for tracking agent actions and decisions"""
    action: str = Field(description="Type of action taken by the agent")
    reasoning: str = Field(description="Reasoning behind the action")
    query: str = Field(description="Query used for retrieval or analysis")
    result: Any = Field(description="Result of the action")


class ComplianceChecklistOutput(BaseModel):
    checklist: dict = Field(description="Dictionory which contains extracted compliance checklist")


class RiskClause(BaseModel):
    """Schema for an identified risk clause"""
    category: str = Field(description="Category of the risk (e.g., Termination, Indemnification)")
    problematic_clause: str = Field(description="Direct quote of the problematic clause from the RFP")
    risk_assessment: str = Field(description="Brief explanation of why this clause is problematic")
    suggested_alternative: str = Field(description="Suggested more balanced or neutral alternative language")

class RiskAnalysisOutput(BaseModel):
    """Schema for the full risk analysis output"""
    identified_risks: List[RiskClause] = Field(description="List of identified risky clauses")
    summary: str = Field(description="Brief summary of overall contract risk profile")

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
    category: str = Field(description="The type of clause (e.g., Termination, Indemnification)")
    clause: str = Field(description="The problematic clause text")
    risk: str = Field(description="Assessment of why the clause is problematic")
    alternative: str = Field(description="Suggested balanced alternative")
    risk_level: str = Field(description="Risk level (Low, Medium, High)")

class RiskAnalysisOutput(BaseModel):
    risk_clauses: List[RiskClause] = Field(description="List of identified risk clauses eg. Unilateral Termination, Indemnity clauses, No Exit clauses")
    overall_risk_level: str = Field(description="Overall risk assessment of the RFP")

class ReasoningStep(BaseModel):
    thought: str = Field(description="The reasoning or thought process")
    action: str = Field(description="The action to take based on the reasoning")
    observation: Optional[str] = Field(None, description="The result of the action")

class RiskAnalysisWithReasoning(RiskAnalysisOutput):
    reasoning_trace: List[ReasoningStep] = Field(description="The step-by-step reasoning process")
    improvement_suggestions: List[str] = Field(description="Suggestions for improving the analysis")


class ReasoningStep(BaseModel):
    thought: str
    action: str
    observation: Optional[str] = None
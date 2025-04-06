import os
import json
from agents.config import config
from agents.eligibility_agent import EligibilityAnalyzerAgent
from agents.complaince_agent import ComplianceChecklistAgent
from agents.contract_risk_agent import ReActRiskClauseAnalyzerAgent
from agents.helper.loading_docs import load_document, split_documents_semantic


def run_rfp_analysis_pipeline(company_data_file_path: str, rfp_path: str, iterations: int = 3):
    """
    Complete RFP analysis pipeline that:
    1. Loads and processes company and RFP documents
    2. Extracts compliance checklist
    3. Analyzes eligibility requirements
    4. Saves results
    
    Args:
        company_data_file_path: Path to company data document
        rfp_path: Path to RFP document
        iterations: Number of iterations for eligibility analysis
    
    Returns:
        tuple: (eligibility_result, compliance_checklist)
    """
    print("🚀 Starting RFP Analysis Pipeline")
    
    # Initialize agents
    compliance_agent = ComplianceChecklistAgent()
    eligibility_agent = EligibilityAnalyzerAgent()
    contract_risk_agent = ReActRiskClauseAnalyzerAgent()

    # Load and process documents
    company_document = load_document(company_data_file_path, "company_document")
    rfp_document = load_document(rfp_path, "rfp_document")
    
    # Split into semantic chunks
    company_chunks = split_documents_semantic(company_document)
    rfp_chunks = split_documents_semantic(rfp_document)

    # Responses
    compliance_agent_response = compliance_agent.extract_compliance_checklist(rfp_path)

    eligibility_agent_response = eligibility_agent.analyze_eligibility_from_chunks(
        company_chunks=company_chunks, 
        rfp_chunks=rfp_chunks, 
        iterations=3
    )

    try:
        if os.path.exists(config.ELIGIBILITY_JSON):
            with open(config.ELIGIBILITY_JSON, "r") as f:
                eligibility_result = json.load(f)

        if os.path.exists(config.COMPLIANCE_CHECKLIST_JSON):
            with open(config.COMPLIANCE_CHECKLIST_JSON, "r") as f:
                compliance_result = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load previous agent outputs: {e}")

    eligibility_result = None

    print("\n🔎 Launching ReAct Risk Clause Analyzer...")
    contract_risk_agent_response = contract_risk_agent.iterative_analysis(
        "./documents/RFPs/ELIGIBLE RFP - 2.pdf",
        eligibility_result=eligibility_result,
        compliance_result=compliance_result
    )


    return compliance_agent_response, eligibility_agent_response, contract_risk_agent_response

response = run_rfp_analysis_pipeline(
    "../documents/company_data/Company Data.docx",
    "../documents/RFPs/ELIGIBLE RFP - 2.pdf"
)

print(response)
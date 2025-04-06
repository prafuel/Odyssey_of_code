import streamlit as st
import json
import os
import tempfile
from datetime import datetime
import time

# Import your agents and helper functions
# Assuming these imports would work in your actual environment
from agents.helper.loading_docs import load_document, split_documents_semantic

from agents.complaince_agent import ComplianceChecklistAgent
from agents.contract_risk_agent import ReActRiskClauseAnalyzerAgent
from agents.eligibility_agent import EligibilityAnalyzerAgent
from agents.config import config

# Set page config
st.set_page_config(
    page_title="RFP Analysis Pipeline",
    page_icon="📋",
    layout="wide"
)

# Define helper functions
def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def run_eligibility_analysis(rfp_path, company_data_path):
    """Run the eligibility analyzer agent and return results"""
    with st.spinner("Running Eligibility Analysis..."):
        eligibility_analyzer_agent = EligibilityAnalyzerAgent()
        
        # Clear previous vector databases
        eligibility_analyzer_agent.clear_vector_stores()
        
        # Load and process documents
        company_document = load_document(company_data_path, "company_document")
        rfp_document = load_document(rfp_path, "rfp_document")
        
        # Split into semantic chunks
        company_chunks = split_documents_semantic(company_document)
        rfp_chunks = split_documents_semantic(rfp_document)
        
        # Analyze eligibility using pre-processed chunks
        result = eligibility_analyzer_agent.analyze_eligibility_from_chunks(
            company_chunks=company_chunks,
            rfp_chunks=rfp_chunks,
            iterations=1
        )
        
        # Save final result
        with open(config.ELIGIBILITY_JSON, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        
        # Save feedback history
        eligibility_analyzer_agent.save_feedback_history(filename=config.FEEDBACK_ELIGIBILITY_JSON)
        
        return result

def run_compliance_analysis(rfp_path):
    """Run the compliance checklist agent and return results"""
    with st.spinner("Running Compliance Analysis..."):
        compliance_agent = ComplianceChecklistAgent()
        
        # Clear previous vector databases
        compliance_agent.clear_vector_stores()
        
        # Analyze compliance checklist for a specific RFP
        result = compliance_agent.extract_compliance_checklist(rfp_path)
        
        # Save results to file
        with open(config.COMPLIANCE_CHECKLIST_JSON, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        
        return result

def run_risk_analysis(rfp_path, eligibility_result=None, compliance_result=None):
    """Run the risk clause analyzer agent and return results"""
    with st.spinner("Running Risk Clause Analysis..."):
        agent = ReActRiskClauseAnalyzerAgent()
        agent.clear_vector_store()
        
        # Run iterative analysis
        result = agent.iterative_analysis(
            rfp_path,
            eligibility_result=eligibility_result,
            compliance_result=compliance_result
        )
        
        # Save final result
        with open(config.CONTRACT_RISK_JSON, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        
        return result

# App interface
st.title("RFP Analysis Pipeline")
st.subheader("Upload documents for comprehensive RFP analysis")

# File uploaders
with st.form("document_upload"):
    col1, col2 = st.columns(2)
    
    with col1:
        rfp_file = st.file_uploader("Upload RFP Document (PDF)", type=["pdf"], key="rfp")
    
    with col2:
        company_data_file = st.file_uploader("Upload Company Data (DOCX)", type=["docx"], key="company")
    
    submit_button = st.form_submit_button("Start Analysis")

# Process files and run analysis when submitted
if submit_button and rfp_file is not None and company_data_file is not None:
    # Save uploaded files
    rfp_path = save_uploaded_file(rfp_file)
    company_data_path = save_uploaded_file(company_data_file)
    
    if rfp_path and company_data_path:
        # Create tabs for each analysis step
        tabs = st.tabs(["Eligibility Analysis", "Compliance Checklist", "Risk Analysis", "Summary"])
        
        # Tab 1: Eligibility Analysis
        with tabs[0]:
            st.header("Eligibility Analysis")
            eligibility_result = run_eligibility_analysis(rfp_path, company_data_path)
            
            # Display eligibility results
            st.success(f"Eligibility Status: {'✅ Eligible' if eligibility_result.eligible else '❌ Not Eligible'}")
            
            st.subheader("Eligibility Reasons")
            for reason in eligibility_result.reasons:
                st.write(f"- {reason}")
            
            if eligibility_result.matching_requirements:
                st.subheader("Requirements Met")
                for req in eligibility_result.matching_requirements:
                    st.write(f"✅ {req}")
            
            if eligibility_result.missing_requirements:
                st.subheader("Requirements Not Met")
                for req in eligibility_result.missing_requirements:
                    st.write(f"❌ {req}")
            
            st.subheader("Recommended Actions")
            for action in eligibility_result.recommended_actions:
                st.write(f"- {action}")
                
        # Tab 2: Compliance Checklist
        with tabs[1]:
            st.header("Compliance Checklist")
            compliance_result = run_compliance_analysis(rfp_path)
            
            # Display compliance results based on your actual model structure
            st.subheader("Compliance Checklist")
            
            # Since you only have a dictionary called 'checklist', display its contents
            # This assumes the checklist has some structure you want to display
            checklist = compliance_result.checklist
            
            # Example of how you might display the checklist
            for category, items in checklist.items():
                st.subheader(category)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and 'name' in item and 'status' in item:
                            status = "✅" if item['status'] else "❌"
                            with st.expander(f"{item['name']} - {status}"):
                                if 'details' in item:
                                    st.write(item['details'])
                        else:
                            st.write(f"- {item}")
                else:
                    st.write(items)
        
        # Tab 3: Risk Analysis
        with tabs[2]:
            st.header("Risk Clause Analysis")
            risk_result = run_risk_analysis(
                rfp_path, 
                eligibility_result=eligibility_result.model_dump(), 
                compliance_result=compliance_result.model_dump()
            )
            
            # Display risk analysis results
            st.success(f"Overall Risk Score: {risk_result.overall_risk_score}/10")
            st.write(risk_result.risk_summary)
            
            st.subheader("Risk Clauses")
            for clause in risk_result.risk_clauses:
                risk_level = "🔴 High" if clause.risk_level == "high" else "🟡 Medium" if clause.risk_level == "medium" else "🟢 Low"
                with st.expander(f"{clause.clause_title} - {risk_level} (Page {clause.page_number})"):
                    st.write(f"**Risk Assessment:** {clause.risk_assessment}")
                    st.write(f"**Recommendation:** {clause.recommendation}")
        
        # Tab 4: Summary
        with tabs[3]:
            st.header("Summary Report")
            
            # Create a summary table
            st.subheader("Key Metrics")
            col2, col3 = st.columns(3)
            with col2:
                st.metric("Compliance Score", f"{compliance_result.compliance_score}/100")
            with col3:
                st.metric("Risk Score", f"{risk_result.overall_risk_score}/10")
            
            # Overall recommendation
            st.subheader("Final Assessment")
            
            # Logic for final recommendation
            if (eligibility_result.eligibility_score > 70 and 
                compliance_result.compliance_score > 70 and 
                risk_result.overall_risk_score < 5):
                recommendation = "Proceed with proposal"
                recommendation_color = "green"
            elif (eligibility_result.eligibility_score < 50 or 
                  compliance_result.compliance_score < 50 or 
                  risk_result.overall_risk_score > 7):
                recommendation = "Do not proceed"
                recommendation_color = "red"
            else:
                recommendation = "Proceed with caution"
                recommendation_color = "orange"
            
            st.markdown(f"<h3 style='color:{recommendation_color}'>{recommendation}</h3>", unsafe_allow_html=True)
            
            # Combined summary
            combined_summary = f"""
            **Eligibility Assessment**: {eligibility_result.eligibility_assessment}
            
            **Compliance Overview**: {compliance_result.compliance_overview}
            
            **Risk Summary**: {risk_result.risk_summary}
            """
            st.markdown(combined_summary)
            
            # Download results
            st.subheader("Download Results")
            
            # Export all results to a combined JSON file
            combined_results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rfp_filename": rfp_file.name,
                "company_data_filename": company_data_file.name,
                "eligibility_result": eligibility_result.model_dump(),
                "compliance_result": compliance_result.model_dump(),
                "risk_result": risk_result.model_dump()
            }
            
            combined_json = json.dumps(combined_results, indent=2)
            st.download_button(
                label="Download Full Analysis (JSON)",
                data=combined_json,
                file_name=f"rfp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

else:
    # Display instructions when no files are uploaded
    st.info("Please upload both an RFP document (PDF) and company data file (DOCX) to begin analysis.")
    
    # Show example of what the app does
    with st.expander("How this app works"):
        st.write("""
        This application analyzes Request for Proposal (RFP) documents through three sequential analysis steps:
        
        1. **Eligibility Analysis**: Evaluates if your company meets the basic requirements to bid on the RFP
        2. **Compliance Checklist**: Identifies required documents, certifications, and submission requirements
        3. **Risk Clause Analysis**: Highlights potentially risky contract clauses and provides recommendations
        
        After uploading your documents, the analysis will run automatically and results will be displayed in separate tabs.
        """)
        
        st.image("https://via.placeholder.com/800x400.png?text=RFP+Analysis+Pipeline+Flow", 
                 caption="Analysis Pipeline Visualization")
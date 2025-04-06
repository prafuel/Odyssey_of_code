import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# from agents.config import config
# from agents.schema import ComplianceChecklistOutput

from config import config
from schema import ComplianceChecklistOutput

from dotenv import load_dotenv
load_dotenv()

class ComplianceChecklistAgent:
    def __init__(self):
        """Initialize the Compliance Checklist Generator Agent"""
        self.config = config
        
        # Create directory for vector database if it doesn't exist
        os.makedirs(self.config.RFP_VD, exist_ok=True)
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)

        # Initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=ComplianceChecklistOutput)
        
        # Initialize LLM
        self.llm = ChatGroq(model_name=config.MAIN_LLM)
        
        # RFP data
        self.rfp_vectorstore = None
        self.rfp_retriever = None
        
    def load_rfp(self, chunks: list):
        # """Load and process RFP document"""
        # print(f"📄 Loading RFP document: {os.path.basename(rfp_path)}...")
        
        # # Load document
        # pages = load_document(rfp_path, "rfp_document")
        
        # # Split RFP into chunks
        # print("✂️ Splitting RFP into manageable chunks...")
        # chunks = split_documents_semantic(pages)
        
        # print("chunk : ", chunks)
        chunks = [Document(chunk) for chunk in chunks]


        # Create vector embeddings using Chroma
        print("🔍 Creating Chroma vector index for RFP content...")
        self.rfp_vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory=self.config.RFP_VD
        )
        
        # Persist the vector database
        self.rfp_vectorstore.persist()
        
        # Create retriever for RFP
        self.rfp_retriever = self.rfp_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        return self.rfp_retriever, chunks
    
    def extract_compliance_checklist(self, chunks: list) -> Dict[str, Any]:
        """Main method to extract compliance checklist from RFP using LCEL"""
        print("chunks: ", chunks)
        # Load and process RFP
        rfp_retriever, _ = self.load_rfp(chunks=chunks)
        
        # Create compliance checklist prompt
        prompt_template = """
            You are an RFP Compliance Analyst AI.
            Your task is to carefully extract all submission requirements and formatting rules from the RFP document provided below. Create a comprehensive, structured checklist that proposal teams can use to ensure full compliance.

            Analyze the document for:

            DOCUMENT_FORMAT_REQUIREMENTS:
            - Page limits (overall and per-section)
            - Font specifications (type, size, color)
            - Margin requirements
            - Line spacing
            - Header/footer specifications
            - Table of Contents requirements
            - Section numbering conventions
            - File format requirements (PDF, Word, etc.)

            SUBMISSION_LOGISTICS:
            - Submission deadline (date and time, including timezone)
            - Submission method (electronic portal, email, physical delivery)
            - Number of copies required
            - Special packaging or labeling instructions

            REQUIRED_COMPONENTS:
            - Mandatory forms and attachments
            - Required certifications or acknowledgments
            - Required signatures and their locations
            - Specific section organization requirements
            - Required table formats
            - Required graphics or visual elements

            DEAL_BREAKERS:
            - Identify any requirements described as "mandatory," "must," "shall," or "required"
            - Note any statements like "failure to comply will result in disqualification"
            - Highlight any sections labeled as "Minimum Requirements" or "Mandatory Requirements"

            Format your response as a structured checklist with clear categories, numbered items, and page references to the original RFP where possible. Flag potential deal breakers with [CRITICAL] to help teams prioritize compliance efforts.

            RFP Content:
            {context}

            {format_instructions}
        """

        # Create LCEL chain
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        compliance_chain = (
            {"context": rfp_retriever}
            | prompt
            | self.llm
            | self.parser
        )
        
        # Run compliance analysis
        print("🔎 Extracting compliance checklist...")
        result = compliance_chain.invoke("Extract the compliance checklist instructions from this RFP")
        
        return result
    
    def clear_vector_stores(self) -> bool:
        """Clear the vector stores for fresh analysis"""
        print("🧹 Clearing previous vector stores...")
        try:
            # Load and delete existing vector store
            if os.path.exists(self.config.RFP_VD):
                rfp_db = Chroma(
                    persist_directory=self.config.RFP_VD, 
                    embedding_function=self.embeddings
                )
                rfp_db.delete_collection()
                
            print("✅ Vector stores cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing vector stores: {e}")
            return False


# Example usage
if __name__ == "__main__":
    compliance_agent = ComplianceChecklistAgent()
    
    # Optional: Clear previous vector databases for fresh analysis
    compliance_agent.clear_vector_stores()
        
    # Analyze compliance checklist for a specific RFP
    result = compliance_agent.extract_compliance_checklist("./documents/RFPs/ELIGIBLE RFP - 1.pdf")
    
    # Print results
    # print("\n📋 COMPLIANCE CHECKLIST RESULT:")
    print(json.dumps(result.model_dump(), indent=2))
    
    # Save results to file
    with open(config.COMPLIANCE_CHECKLIST_JSON, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    print("\n✅ Checklist saved to compliance_checklist.json")
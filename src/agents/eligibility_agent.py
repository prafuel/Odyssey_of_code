import os
from typing import List
from langchain_groq import ChatGroq
from schema import EligibilityAgentOutput
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader

from config import config
from schema import EligibilityAgentOutput

from dotenv import load_dotenv
load_dotenv()

class EligibilityAnalyzerAgent:
    def __init__(self):
        """Initialize the Eligibility Analyzer Agent"""
        self.config = config
        
        # Create directories for vector databases if they don't exist
        os.makedirs(self.config.COMPANY_VD, exist_ok=True)
        os.makedirs(self.config.RFP_VD, exist_ok=True)
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)

        self.parser = PydanticOutputParser(pydantic_object=EligibilityAgentOutput)
        
        # Initialize LLM
        self.llm = ChatGroq(model_name=config.MAIN_LLM)
        
        # Company profile data
        self.company_vectorstore = None
        self.company_retriever = None
        
    def load_company_profile(self, company_profile_path: str) -> bool:
        """Load and process company profile from a DOCX file"""
        print("📋 Loading company profile...")
        
        # Load document
        loader = Docx2txtLoader(company_profile_path)
        documents = loader.load()
        
        # Split company profile into chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        # Create vector embeddings using Chroma
        print("🔍 Creating Chroma vector index for company profile...")
        self.company_vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory=self.config.COMPANY_VD
        )
        
        # Persist the vector database
        self.company_vectorstore.persist()
        
        # Create retriever for company information
        self.company_retriever = self.company_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        print("✅ Company profile loaded and indexed successfully")
        return True
    
    def load_rfp(self, rfp_path: str):
        """Load and process RFP document"""
        print(f"📄 Loading RFP document: {os.path.basename(rfp_path)}...")
        
        # Load document
        loader = PyMuPDFLoader(rfp_path)
        pages = loader.load()
        
        # Split RFP into chunks
        print("✂️ Splitting RFP into manageable chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        
        # Create vector embeddings using Chroma
        print("🔍 Creating Chroma vector index for RFP content...")
        rfp_vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory=self.config.RFP_VD
        )
        
        # Persist the vector database
        rfp_vectorstore.persist()
        
        # Create retriever for RFP
        rfp_retriever = rfp_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        return rfp_retriever, chunks
    
    def get_relevant_company_info(self, rfp_chunks: List) -> str:
        """Extract company information relevant to the RFP requirements"""
        if not self.company_retriever:
            raise ValueError("Company profile not loaded. Call load_company_profile first.")
        
        # Extract key requirements from RFP
        requirements = []
        
        # Look at first few chunks for requirements
        for chunk in rfp_chunks[:5]:
            # Get relevant company info based on this RFP chunk
            docs = self.company_retriever.invoke(
                f"Find company information related to: {chunk.page_content[:300]}"
            )
            for doc in docs:
                requirements.append(doc.page_content)
        
        # Deduplicate and join
        unique_requirements = list(set(requirements))
        return "\n".join(unique_requirements)
    
    def analyze_eligibility(self, rfp_path: str) -> EligibilityAgentOutput:
        """Main method to analyze if company is eligible for the RFP using LCEL"""
        if not self.company_retriever:
            raise ValueError("Company profile not loaded. Call load_company_profile first.")
        
        # Load and process RFP
        rfp_retriever, rfp_chunks = self.load_rfp(rfp_path)
        
        # Get relevant company information for this specific RFP
        relevant_company_info = self.get_relevant_company_info(rfp_chunks)
        
        # Create eligibility analyzer prompt
        prompt_template = """
            You are RFP Eligibility Analyzer.
            
            Your task is to determine if the company meets the eligibility criteria for this RFP by strictly comparing the RFP requirements against the company's qualifications.
            
            RFP Context (requirements and eligibility criteria):
            {context}
            
            Company's Relevant Qualifications:
            {company_data}
            
            Based on the above information, evaluate if the company meets the following criteria also include RFP page number if possible for better user understanding:
            1. Required certifications and accreditations
            2. State/federal registrations and legal status
            3. Required past experience and project history
            4. Technical capabilities required by the RFP
            5. Compliance with any special requirements (small business, etc.)
            
            Your response must be a valid JSON object with this structure:
            format_instructions={format_instructions}
        """

        def output(x):
            return x.content

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "company_data"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # Create LCEL chain
        eligibility_chain = (
            RunnableParallel(
                context=rfp_retriever,
                company_data=lambda _: relevant_company_info
            )
            | prompt
            | self.llm
            | self.parser
        )
        
        # Run eligibility analysis
        print("🔎 Analyzing eligibility criteria...")
        result = eligibility_chain.invoke("Is the company eligible for this RFP based on the requirements?")
        
        return result
    
    def clear_vector_stores(self) -> bool:
        """Clear the vector stores for fresh analysis"""
        print("🧹 Clearing previous vector stores...")
        try:
            # Load and delete existing vector stores
            if os.path.exists(self.config.COMPANY_VD):
                company_db = Chroma(
                    persist_directory=self.config.COMPANY_VD, 
                    embedding_function=self.embeddings
                )
                company_db.delete_collection()
            
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
    eligibility_analyzer_agent = EligibilityAnalyzerAgent()
    
    # Optional: Clear previous vector databases for fresh analysis
    eligibility_analyzer_agent.clear_vector_stores()
    
    # Load company profile
    eligibility_analyzer_agent.load_company_profile("./documents/company_data/Company Data.docx")
    
    # Analyze eligibility for a specific RFP
    result = eligibility_analyzer_agent.analyze_eligibility("./documents/RFPs/ELIGIBLE RFP - 1.pdf")
    
    # Print results
    # print("\n📊 ELIGIBILITY ANALYSIS RESULT:")
    # print(json.dumps(result.model_dump(), indent=2))
    print(result)

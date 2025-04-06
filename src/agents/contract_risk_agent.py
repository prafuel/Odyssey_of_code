import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from config import config
from schema import RiskClause, RiskAnalysisOutput
from dotenv import load_dotenv

load_dotenv()

class RiskClauseAnalyzerAgent:
    def __init__(self):
        self.RISK_VD = config.RFP_VD
        self.EMBEDDING_MODEL = config.EMBEDDING_MODEL
        self.MAIN_LLM = config.MAIN_LLM

        os.makedirs(self.RISK_VD, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        self.parser = PydanticOutputParser(pydantic_object=RiskAnalysisOutput)
        self.llm = ChatGroq(model_name=self.MAIN_LLM)

    def load_rfp(self, rfp_path: str):
        print(f"📄 Loading RFP: {os.path.basename(rfp_path)}")

        loader = PyMuPDFLoader(rfp_path)
        pages = loader.load()

        print("✂️ Splitting RFP into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        print("🔍 Indexing RFP with Chroma DB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.RISK_VD
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        return retriever

    def analyze_risk_clauses(self, rfp_path: str, eligibility_result=None, compliance_result=None) -> Dict[str, Any]:
        retriever = self.load_rfp(rfp_path)

        # Load context from previous agents if available
        additional_context = ""
        if eligibility_result:
            additional_context += f"\nEligibility Analysis:\n{json.dumps(eligibility_result, indent=2)}"
        if compliance_result:
            additional_context += f"\nCompliance Checklist:\n{json.dumps(compliance_result, indent=2)}"
        if not additional_context:
            additional_context = "No prior agent context available."

        reference_examples = """
Category: Termination
Problematic: "The client may terminate this agreement at any time without notice."
Risk: This clause allows the client to cancel the contract unilaterally, posing a high risk for the vendor.
Alternative: "Either party may terminate the agreement with a 30-day written notice."

Category: Indemnification
Problematic: "The vendor shall indemnify the client for any and all losses."
Risk: This implies unlimited liability without defining scope or fault.
Alternative: "The vendor shall indemnify the client for direct losses arising from proven negligence."
"""

        prompt_template = """
You are a legal risk analysis expert.

Your task is to analyze the provided RFP for contractual clauses that may pose a risk to the vendor.

Identify:
- The clause category (Termination, Indemnification, Exit Clause, IP Rights, Payment Terms)
- A direct quote of the problematic clause
- A short explanation of why it's problematic
- A more balanced alternative clause

Provide your output in this structured JSON format:
{format_instructions}

RFP Content:
{context}

Reference examples of problematic vs. balanced clauses:
{reference_examples}

Additional Context from Previous Analysis:
{additional_context}

End with a summary of the overall risk level of this RFP.
"""

        rag_prompt = PromptTemplate(
            input_variables=["context", "reference_examples", "additional_context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template=prompt_template
        )

        print("🤖 Running RAG-based risk clause analysis...")

        rag_chain = (
            retriever
            | (lambda docs: {"context": "\n\n".join([doc.page_content for doc in docs])})
            | RunnableMap({
                "context": lambda x: x["context"],
                "reference_examples": lambda _: reference_examples,
                "additional_context": lambda _: additional_context,
            })
            | rag_prompt
            | self.llm
            | self.parser
        )

        response = rag_chain.invoke("Find risky contract clauses in this RFP and suggest neutral alternatives.")
        return response

    def clear_vector_store(self) -> bool:
        print("🧹 Clearing old vector store...")
        try:
            if os.path.exists(self.RISK_VD):
                vector_db = Chroma(persist_directory=self.RISK_VD, embedding_function=self.embeddings)
                vector_db.delete_collection()
            print("✅ Cleared risk vector DB.")
            return True
        except Exception as e:
            print(f"❌ Error clearing vector DB: {e}")
            return False

if __name__ == "__main__":
    agent = RiskClauseAnalyzerAgent()

    agent.clear_vector_store()

    eligibility_result = None
    compliance_result = None

    try:
        if os.path.exists("eligibility_result.json"):
            with open("eligibility_result.json", "r") as f:
                eligibility_result = json.load(f)

        if os.path.exists("compliance_checklist.json"):
            with open("compliance_checklist.json", "r") as f:
                compliance_result = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load previous agent outputs: {e}")

    print("\n🔎 Launching Risk Clause Analyzer...")
    result = agent.analyze_risk_clauses(
        "./documents/RFPs/ELIGIBLE RFP - 1.pdf",
        eligibility_result=eligibility_result,
        compliance_result=compliance_result
    )

    print("\n📊 RISK ANALYSIS OUTPUT:")
    print(json.dumps(result.dict(), indent=2))

    with open("risk_analysis.json", "w") as f:
        json.dump(result.dict(), f, indent=2)
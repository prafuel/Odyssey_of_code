import os
import json
import time
from langchain_groq import ChatGroq
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser

from config import config
from schema import EligibilityAgentOutput

# from agents.config import config
# from agents.schema import EligibilityAgentOutput


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
        
        # RFP data
        self.rfp_vectorstore = None
        self.rfp_retriever = None
        
        # Feedback history to improve responses
        self.feedback_history = []
    
    def process_document_chunks(self, document_chunks: List, doc_type: str) -> Any:
        """Process pre-split document chunks and create vector store"""
        print(f"🔍 Creating Chroma vector index for {doc_type} content...")
        
        # Select the appropriate directory based on document type
        persist_dir = self.config.COMPANY_VD if doc_type == "company_document" else self.config.RFP_VD
        
        vectorstore = Chroma.from_documents(
            documents=document_chunks, 
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        # Persist the vector database
        vectorstore.persist()
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5 if doc_type == "company_document" else 8}
        )
        
        # Store references based on document type
        if doc_type == "company_document":
            self.company_vectorstore = vectorstore
            self.company_retriever = retriever
        else:
            self.rfp_vectorstore = vectorstore
            self.rfp_retriever = retriever
        
        print(f"✅ {doc_type} indexed successfully")
        return retriever
    
    def process_company_chunks(self, company_chunks: List) -> bool:
        """Process pre-split company document chunks"""
        self.process_document_chunks(company_chunks, "company_document")
        return True
    
    def process_rfp_chunks(self, rfp_chunks: List) -> Any:
        """Process pre-split RFP document chunks"""
        return self.process_document_chunks(rfp_chunks, "rfp_document"), rfp_chunks
    
    def get_relevant_company_info(self, rfp_chunks: List) -> str:
        """Extract company information relevant to the RFP requirements"""
        if not self.company_retriever:
            raise ValueError("Company profile not loaded. Call process_company_chunks first.")
        
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
    
    def extract_rfp_requirements(self, rfp_chunks: List) -> Dict[str, Any]:
        """Use ReAct approach to extract key RFP requirements"""
        print("🔎 Extracting RFP requirements using ReAct approach...")
        
        # ReAct prompt to extract requirements
        react_extract_prompt = """You are a specialized RFP Analyzer that extracts key requirements from RFP documents.

        # Task
        Extract the most critical eligibility requirements from the provided RFP sections.

        # RFP Content
        {rfp_content}

        # ReAct Format
        Think through this step-by-step:
        1. Thought: First, analyze what types of requirements I should look for
        2. Action: Identify specific eligibility criteria in the RFP sections
        3. Observation: Note exactly what I found and where (page/section)
        4. Thought: Consider if these are mandatory or preferred requirements
        5. Action: Categorize requirements by type (certifications, experience, etc.)

        Using this approach, extract and organize the requirements:
        """
        
        # Combine chunks for analysis
        rfp_combined = "\n\n".join([chunk.page_content for chunk in rfp_chunks[:10]])
        
        # Extract requirements using LLM
        response = self.llm.invoke(react_extract_prompt.format(rfp_content=rfp_combined))
        
        # Structure the response
        categories = [
            "certifications", "registrations", "experience", 
            "technical_capabilities", "special_requirements"
        ]
        
        requirements = {
            "raw_extraction": response.content,
            "categorized": {}
        }
        
        # Secondary prompt to structure the requirements
        structure_prompt = f"""Given the following extracted RFP requirements:
        
        {response.content}
        
        Please categorize these requirements into the following categories:
        1. certifications: [list required certifications]
        2. registrations: [list required registrations and legal status]
        3. experience: [list required experience and project history]
        4. technical_capabilities: [list technical capabilities required]
        5. special_requirements: [list any special requirements]
        
        Provide the response as a JSON object with these 5 categories as keys.
        """
        
        structured_response = self.llm.invoke(structure_prompt)
        
        # Extract JSON from response
        try:
            # Find JSON in response
            json_start = structured_response.content.find("{")
            json_end = structured_response.content.rfind("}") + 1
            json_str = structured_response.content[json_start:json_end]
            
            # Parse JSON
            structured_requirements = json.loads(json_str)
            requirements["categorized"] = structured_requirements
        except:
            print("⚠️ Could not parse JSON from structured response, using raw extraction")
            # Create empty categories
            requirements["categorized"] = {cat: [] for cat in categories}
        
        return requirements
        
    def analyze_eligibility_from_chunks(self, company_chunks: List, rfp_chunks: List, iterations: int = 3) -> EligibilityAgentOutput:
        """Main method to analyze eligibility using pre-processed document chunks"""
        # Process company chunks
        self.process_company_chunks(company_chunks)
        
        # Process RFP chunks
        rfp_retriever, _ = self.process_rfp_chunks(rfp_chunks)
        
        # Get relevant company information for this specific RFP
        relevant_company_info = self.get_relevant_company_info(rfp_chunks)
        
        # Extract requirements using ReAct
        requirements = self.extract_rfp_requirements(rfp_chunks)
        
        # Initialize best result and score
        best_result = None
        best_score = 0
        
        # Initialize result variable to avoid UnboundLocalError
        result = None
        
        # Feedback-driven iteration loop
        for iteration in range(iterations):
            print(f"\n🔄 Running analysis iteration {iteration + 1}/{iterations}")
            
            # Create react prompt with feedback history incorporated
            react_prompt = self.create_react_prompt(
                requirements=requirements, 
                company_data=relevant_company_info,
                iteration=iteration
            )
            
            # Run eligibility analysis with ReAct
            try:
                result = self.run_react_analysis(
                    rfp_retriever=rfp_retriever,
                    react_prompt=react_prompt,
                    requirements=requirements,
                    company_data=relevant_company_info
                )
                
                # Generate self-critique and score
                critique, score = self.generate_self_critique(result, rfp_chunks, relevant_company_info)
                
                print(f"✏️ Self-critique score: {score}/10")
                
                # Store feedback for next iteration
                self.feedback_history.append({
                    "iteration": iteration + 1,
                    "result": result.model_dump(),
                    "critique": critique,
                    "score": score
                })
                
                # Update best result if current is better
                if score > best_score:
                    best_result = result
                    best_score = score
                
                # Allow some time between iterations to avoid rate limiting
                if iteration < iterations - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Error in iteration {iteration + 1}: {e}")
                continue
        
        # Handle case where all iterations failed
        if best_result is None and result is None:
            raise ValueError("All analysis iterations failed. Check logs for details.")
                
        # Return best result or latest if no successful iterations
        return best_result if best_result else result
    
    def create_react_prompt(self, requirements: Dict, company_data: str, iteration: int) -> str:
        """Create ReAct prompt incorporating feedback from previous iterations"""
        
        # Base ReAct prompt
        base_prompt = """You are RFP Eligibility Analyzer using the ReAct (Reasoning + Acting) approach.
                # Task
                Determine if the company meets the eligibility criteria for this RFP by comparing the RFP requirements against the company's qualifications.

                # RFP Requirements
                {requirements_text}

                # Company's Qualifications
                {company_data}

                # ReAct Format
                Follow this step-by-step process:

                1. Thought: [Think about what criteria needs to be evaluated]
                2. Action: [Look for specific criteria in the requirements]
                3. Observation: [Note what you found or didn't find about this criteria]
                4. Thought: [Analyze if the company meets this requirement]
                5. Action: [Look for evidence of this capability in company data]
                6. Observation: [Note what evidence you found or didn't find]

                Repeat this process for each of these categories:
                - Required certifications and accreditations
                - State/federal registrations and legal status
                - Required past experience and project history
                - Technical capabilities required by the RFP
                - Compliance with any special requirements

                After your analysis, provide a structured output following this format:
                {format_instructions}
        """
        
        # Add feedback from previous iterations if available
        feedback_section = ""
        if iteration > 0 and self.feedback_history:
            feedback_section = "\n# Previous Analysis Feedback\nConsider these critiques of previous analyses:\n"
            for fb in self.feedback_history[-min(3, len(self.feedback_history)):]:
                feedback_section += f"\nIteration {fb['iteration']} critique:\n{fb['critique']}\n"
                
        # Format requirements as text - pre-format this to avoid template variable conflicts
        req_text = json.dumps(requirements["categorized"], indent=2)
        
        full_prompt = base_prompt + feedback_section
        # Note the variable name change from 'requirements' to 'requirements_text'
        return full_prompt.format(
            requirements_text=req_text,
            company_data=company_data,
            format_instructions=self.parser.get_format_instructions()
        )
    
    def run_react_analysis(self, rfp_retriever, react_prompt, requirements, company_data):
        """Run the ReAct analysis using the prompt and return parsed result"""
        # Create a prompt template directly from the already formatted prompt string
        # This avoids template variable conflicts with JSON
        
        # Run the retriever separately to get context
        context_docs = rfp_retriever.invoke("Get relevant RFP requirements")
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Create a simple chain that just runs the LLM with the full prompt
        # This bypasses the templating issues completely
        try:
            # Direct LLM call with the pre-formatted prompt
            llm_response = self.llm.invoke(react_prompt)
            
            # Parse the response
            result = self.parser.parse(llm_response.content)
            return result
        except Exception as e:
            print(f"❌ Error in LLM processing: {e}")
            # Fall back to a simpler prompt if parsing fails
            fallback_prompt = f"""Analyze if the company meets the eligibility criteria for this RFP.
            
            RFP Context:
            {context[:2000]}
            
            Company Information:
            {company_data[:2000]}
            
            Provide your analysis as a JSON with these fields:
            {self.parser.get_format_instructions()}
            """
            
            llm_response = self.llm.invoke(fallback_prompt)
            result = self.parser.parse(llm_response.content)
            return result

    def generate_self_critique(self, result, rfp_chunks, company_data):
        """Generate self-critique and score for the analysis result"""
        # Combine chunks for context
        rfp_sample = "\n\n".join([chunk.page_content for chunk in rfp_chunks[:3]])
        
        critique_prompt = f"""You are a critical evaluator of RFP eligibility analyses.

        Evaluate the following eligibility analysis for accuracy, completeness, and soundness of reasoning:
        
        Result:
        {json.dumps(result.model_dump(), indent=2)}
        
        RFP Context Sample:
        {rfp_sample}
        
        Company Information Sample:
        {company_data[:1000]}
        
        Critique the analysis on these aspects:
        1. Did it identify all key requirements correctly?
        2. Did it properly evaluate company's qualifications against each requirement?
        3. Is the reasoning sound and evidence-based?
        4. Are the eligibility conclusions justified?
        5. Are the recommended actions practical and specific?
        
        Provide a detailed critique and then score the analysis on a scale of 1-10.
        
        Output format:
        CRITIQUE: [Your detailed critique]
        SCORE: [1-10]
        """
        
        response = self.llm.invoke(critique_prompt)
        
        # Extract score from response
        score = 5  # Default score
        critique = response.content
        
        try:
            score_idx = response.content.find("SCORE:")
            if score_idx != -1:
                score_text = response.content[score_idx:].split("\n")[0]
                score = int(score_text.replace("SCORE:", "").strip())
        except:
            pass
            
        return critique, score
    
    def analyze_eligibility(self, rfp_chunks: List, company_chunks: List) -> EligibilityAgentOutput:
        """Simplified method to analyze eligibility from pre-processed chunks"""
        return self.analyze_eligibility_from_chunks(company_chunks, rfp_chunks, iterations=1)
    
    def extract_eligibility_requirements(self, rfp_chunks: List) -> Dict[str, Any]:
        """Wrapper method to extract RFP requirements for external use"""
        return self.extract_rfp_requirements(rfp_chunks)
    
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
    
    def save_feedback_history(self, filename: str = "feedback_history.json"):
        """Save feedback history to a file for analysis"""
        with open(filename, "w") as f:
            json.dump(self.feedback_history, f, indent=2)
        print(f"✅ Feedback history saved to {filename}")


# Example usage
if __name__ == "__main__":
    from helper.loading_docs import load_document, split_documents_semantic
    
    eligibility_analyzer_agent = EligibilityAnalyzerAgent()
    
    # Optional: Clear previous vector databases for fresh analysis
    eligibility_analyzer_agent.clear_vector_stores()
    
    # Load and process documents
    company_document = load_document("./documents/company_data/Company Data.docx", "company_document")
    rfp_document = load_document("./documents/RFPs/IN-ELIGIBLE_RFP.pdf", "rfp_document")
    
    # Split into semantic chunks
    company_chunks = split_documents_semantic(company_document)
    rfp_chunks = split_documents_semantic(rfp_document)
    
    # Analyze eligibility using pre-processed chunks
    result = eligibility_analyzer_agent.analyze_eligibility_from_chunks(
        company_chunks=company_chunks, 
        rfp_chunks=rfp_chunks, 
        iterations=1
    )
    
    # Print results
    print("\n📊 ELIGIBILITY ANALYSIS RESULT:")
    print(json.dumps(result.model_dump(), indent=2))

    print("\nRecommended Actions:")
    for action in result.recommended_actions:
        print(f"  - {action}")

    # Save final result
    with open(config.ELIGIBILITY_JSON, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    # Save feedback history
    eligibility_analyzer_agent.save_feedback_history(filename=config.FEEDBACK_ELIGIBILITY_JSON)
import os
import json
import time
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from typing import List
from dotenv import load_dotenv

from config import config
from schema import RiskAnalysisWithReasoning, RiskAnalysisOutput, RiskClause, ReasoningStep

# from agents.config import config
# from agents.schema import RiskAnalysisWithReasoning, RiskAnalysisOutput, RiskClause, ReasoningStep



load_dotenv()

class ReActRiskClauseAnalyzerAgent:
    def __init__(self):
        self.RISK_VD = config.RFP_VD
        self.EMBEDDING_MODEL = config.EMBEDDING_MODEL
        self.MAIN_LLM = config.MAIN_LLM
        self.MAX_ITERATIONS = 3
        self.FEEDBACK_HISTORY_FILE = config.RISK_ANALYSIS_JSON

        os.makedirs(self.RISK_VD, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        self.llm = ChatGroq(model_name=self.MAIN_LLM)

        # self.react_parser = PydanticOutputParser(pydantic_object=RiskAnalysisWithReasoning)
        self.parser = PydanticOutputParser(pydantic_object=RiskAnalysisWithReasoning)
        
        # Load feedback history if exists
        self.feedback_history = self._load_feedback_history()

    def _load_feedback_history(self) -> List[Dict]:
        """Load feedback history from file if it exists"""
        if os.path.exists(self.FEEDBACK_HISTORY_FILE):
            try:
                with open(self.FEEDBACK_HISTORY_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback history: {e}")
                return []
        return []

    def _save_feedback_history(self):
        """Save feedback history to file"""
        with open(self.FEEDBACK_HISTORY_FILE, "w") as f:
            json.dump(self.feedback_history, f, indent=2)

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

    def _get_previous_analysis_summary(self, rfp_path: str) -> str:
        """Get summary of previous analyses for the same RFP to guide improvement"""
        rfp_name = os.path.basename(rfp_path)
        relevant_feedback = [item for item in self.feedback_history 
                            if item.get("rfp_path") == rfp_name]
        
        if not relevant_feedback:
            return "No previous analysis history for this RFP."
        
        summary = "PREVIOUS ANALYSIS HISTORY:\n\n"
        for idx, feedback in enumerate(relevant_feedback[-3:], 1):  # Get last 3 entries
            summary += f"Analysis #{idx}:\n"
            summary += f"- Key findings: {', '.join(feedback.get('key_findings', []))}\n"
            summary += f"- Improvement areas: {', '.join(feedback.get('improvement_areas', []))}\n"
            summary += f"- Feedback score: {feedback.get('feedback_score', 'N/A')}/10\n\n"
        
        return summary
    
    def _parse_llm_response_to_pydantic(self, text: str) -> RiskAnalysisWithReasoning:
        """Helper function to parse LLM text response to Pydantic model"""
        try:
            # First try to parse directly if it's already valid JSON
            try:
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].strip()
                
                data = json.loads(json_str)
                return RiskAnalysisWithReasoning(**data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse direct JSON: {e}")
                
            # If direct parsing fails, attempt to extract structured data
            result = RiskAnalysisWithReasoning(
                risk_clauses=[],
                overall_risk_level="Medium",
                reasoning_trace=[],
                improvement_suggestions=[]
            )
            
            # Extract reasoning trace steps
            reasoning_steps = []
            reasoning_pattern = r"\*\*Thought:\*\* (.*?)\n\*\*Action:\*\* (.*?)\n\*\*Observation:\*\* (.*?)(?=\n\n|\Z)"
            import re
            reasoning_matches = re.finditer(reasoning_pattern, text, re.DOTALL)
            
            for match in reasoning_matches:
                step = ReasoningStep(
                thought=match.group(1).strip(),
                action=match.group(2).strip(),
                observation=match.group(3).strip() if match.group(3) else None
            )
                reasoning_steps.append(step)
                
            result.reasoning_trace = reasoning_steps
            
            # Extract risk clauses
            risk_clauses = []
            clause_pattern = r"\*\*Category:\*\* (.*?)\n\*\*Problematic Clause:\*\* (.*?)\n\*\*Risk Assessment:\*\* (.*?)\n\*\*Suggested Alternative:\*\* (.*?)\n\n\*\*Risk Level:\*\* (.*?)(?=\n\n|\Z)"
            clause_matches = re.finditer(clause_pattern, text, re.DOTALL)
            
            for match in clause_matches:
                clause = RiskClause(
                    category=match.group(1).strip(),
                    clause=match.group(2).strip(),
                    risk=match.group(3).strip(),
                    alternative=match.group(4).strip(),
                    risk_level=match.group(5).strip()
                )
                risk_clauses.append(clause)
                
            result.risk_clauses = risk_clauses
            
            # Extract overall risk level
            risk_level_match = re.search(r"overall risk level.*?(Low|Medium|High|Critical)", text, re.IGNORECASE)
            if risk_level_match:
                result.overall_risk_level = risk_level_match.group(1).strip()
                
            # Extract improvement suggestions
            suggestions = []
            if "Improvement Suggestions:" in text:
                suggestions_text = text.split("Improvement Suggestions:")[1].split("Self-Critique:")[0].strip()
                for line in suggestions_text.split("\n"):
                    if line.strip().startswith("-") or line.strip().startswith("*"):
                        suggestions.append(line.strip()[1:].strip())
                    elif "suggest" in line.lower() or "recommend" in line.lower():
                        suggestions.append(line.strip())
            
            if suggestions:
                result.improvement_suggestions = suggestions
                
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Return a minimal valid object if parsing fails
            return RiskAnalysisWithReasoning(
                risk_clauses=[
                    RiskClause(
                        category="Parsing Error",
                        clause="Failed to parse output",
                        risk="The LLM output could not be properly parsed",
                        alternative="Try adjusting the prompt format",
                        risk_level="High"
                    )
                ],
                overall_risk_level="High",
                reasoning_trace=[
                    ReasoningStep(
                        thought="Parsing error occurred",
                        action="Return minimal valid object",
                        observation="Original text: " + text[:100] + "..."
                    )
                ],
                improvement_suggestions=["Fix parsing logic", "Adjust prompt to get better structured output"]
            )

    def analyze_risk_clauses(self, rfp_path: str, eligibility_result=None, compliance_result=None, 
                            iteration=1, previous_result=None) -> RiskAnalysisWithReasoning:
        retriever = self.load_rfp(rfp_path)

        # Load context from previous agents if available
        additional_context = ""
        if eligibility_result:
            additional_context += f"\nEligibility Analysis:\n{json.dumps(eligibility_result, indent=2)}"
        if compliance_result:
            additional_context += f"\nCompliance Checklist:\n{json.dumps(compliance_result, indent=2)}"
        if not additional_context:
            additional_context = "No prior agent context available."

        # Add feedback from previous iterations
        previous_analysis = ""
        if previous_result:
            previous_analysis = f"\nPREVIOUS ITERATION ANALYSIS:\n{json.dumps(previous_result.model_dump(), indent=2)}"

        # Add historical feedback
        historical_feedback = self._get_previous_analysis_summary(rfp_path)

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

        react_prompt_template = """
            You are a legal risk analysis expert using ReAct (Reasoning and Acting) methodology to analyze RFPs.

            Your task is to analyze the provided RFP for contractual clauses that may pose a risk to the vendor.

            Current iteration: {iteration}/{max_iterations}

            First, THINK carefully about how to approach this task. Consider what specific risks to look for based on context.
            Then, specify what ACTION you'll take to identify those risks.
            Finally, record your OBSERVATION after taking that action.

            Repeat this process for each major risk category or section of the RFP.

            Identify:
            - The clause category (Termination, Indemnification, Exit Clause, IP Rights, Payment Terms, etc.)
            - A direct quote of the problematic clause
            - A detailed explanation of why it's problematic
            - A more balanced alternative clause
            - Estimated risk level (Low, Medium, High) with justification

            {previous_iteration_instructions}

            format_instructions={format_instructions}

            RFP Content:
            {context}

            Reference examples of problematic vs. balanced clauses:
            {reference_examples}

            Additional Context from Previous Analysis:
            {additional_context}

            {previous_analysis}

            Historical Analysis Feedback:
            {historical_feedback}

            End with:
            1. Specific improvement suggestions for the next iteration
            
            """

        previous_iteration_instructions = ""
        if iteration > 1:
            previous_iteration_instructions = """
                For this iteration, focus on:
                1. Finding risks you might have missed in the previous iteration
                2. Providing more detailed alternatives to problematic clauses
                3. Considering interactions between different clauses that might compound risks
                4. Refining your risk assessments based on previous observations
            """

        rag_prompt = PromptTemplate(
            input_variables=["context", "reference_examples", "additional_context", 
                        "previous_analysis", "historical_feedback", "iteration", 
                        "previous_iteration_instructions"],
            partial_variables={
                "max_iterations": self.MAX_ITERATIONS,
                "format_instructions": self.parser.get_format_instructions()
            },
            template=react_prompt_template
    )

        print(f"🤖 Running RAG-based risk clause analysis (Iteration {iteration}/{self.MAX_ITERATIONS})...")

        # First use a string output parser to get the raw text
        rag_chain = (
            retriever
            | (lambda docs: {"context": "\n\n".join([doc.page_content for doc in docs])})
            | RunnableMap({
                "context": lambda x: x["context"],
                "reference_examples": lambda _: reference_examples,
                "additional_context": lambda _: additional_context,
                "previous_analysis": lambda _: previous_analysis,
                "historical_feedback": lambda _: historical_feedback,
                "iteration": lambda _: iteration,
                "previous_iteration_instructions": lambda _: previous_iteration_instructions,
            })
            | rag_prompt
            | self.llm
            | (lambda x: x.content)
        )

        raw_response = rag_chain.invoke("Find risky contract clauses in this RFP and suggest neutral alternatives.")
        
        # Parse the response into our Pydantic model
        try:
            # First attempt direct parsing if output is valid JSON
            if raw_response.strip().startswith("{") and raw_response.strip().endswith("}"):
                try:
                    data = json.loads(raw_response)
                    response = RiskAnalysisWithReasoning(**data)
                except:
                    response = self._parse_llm_response_to_pydantic(raw_response)
            else:
                # Try to extract JSON from markdown code block
                import re
                json_match = re.search(r"```json\n(.*?)\n```", raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        data = json.loads(json_str)
                        response = RiskAnalysisWithReasoning(**data)
                    except:
                        response = self._parse_llm_response_to_pydantic(raw_response)
                else:
                    response = self._parse_llm_response_to_pydantic(raw_response)
        except Exception as e:
            print(f"❌ Error parsing response: {e}")
            print(f"Raw response: {raw_response[:500]}...")
            # Create a minimal valid response
            response = RiskAnalysisWithReasoning(
                risk_clauses=[
                    RiskClause(
                        category="Parsing Error",
                        clause="Failed to parse output",
                        risk="The LLM output could not be properly parsed",
                        alternative="Try adjusting the prompt format",
                        risk_level="High"
                    )
                ],
                overall_risk_level="High",
                reasoning_trace=[
                    ReasoningStep(
                        thought="Parsing error occurred",
                        action="Return minimal valid object",
                        observation="Original text: " + raw_response[:100] + "..."
                    )
                ],
                improvement_suggestions=["Fix parsing logic", "Adjust prompt to get better structured output"]
            )
            
        return response

    def evaluate_analysis(self, current_result, previous_result=None) -> Dict:
        """Evaluate the current analysis against the previous one to provide feedback"""
        if previous_result is None:
            return {
                "improvement_score": None,
                "feedback": "First iteration - no comparison available.",
                "key_improvements": [],
                "areas_to_focus": ["Ensure comprehensive coverage of all risk categories", 
                                 "Provide detailed alternatives for high-risk clauses"]
            }
        
        eval_prompt = f"""
            You are evaluating the improvement between two iterations of a risk clause analysis.

            Previous analysis:
            {json.dumps(previous_result.dict(), indent=2)}

            Current analysis:
            {json.dumps(current_result.dict(), indent=2)}

            Please evaluate the improvement on a scale of 1-10, where 10 is perfect:
            1. Comprehensiveness: Did the new analysis identify more relevant risks?
            2. Detail quality: Are explanations and alternatives more specific and actionable?
            3. Risk assessment: Is the risk level assessment better justified?
            4. Overall improvement: How much better is the current analysis overall?

            Provide specific examples of improvements and areas still needing focus.
            
            Format your response as a valid JSON with the following structure:
            {{
                "improvement_score": 7.5,
                "feedback": "Detailed feedback here...",
                "key_improvements": ["Improvement 1", "Improvement 2"],
                "areas_to_focus": ["Area 1", "Area 2"]
            }}
            """
        
        eval_response = self.llm.invoke(eval_prompt)
        
        # Parse feedback with better error handling
        try:
            # Try to parse direct JSON first
            if eval_response.content.strip().startswith("{"):
                feedback_data = json.loads(eval_response.content)
                return feedback_data
            
            # Look for JSON code blocks
            import re
            json_match = re.search(r"```json\n(.*?)\n```", eval_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                feedback_data = json.loads(json_str)
                return feedback_data
                
            # Fall back to line parsing
            eval_lines = eval_response.content.split('\n')
            improvement_score = None
            feedback = eval_response.content
            key_improvements = []
            areas_to_focus = []
            
            for line in eval_lines:
                if "overall improvement" in line.lower() and ":" in line:
                    try:
                        score_text = line.split(":")[-1].strip()
                        if "/" in score_text:
                            improvement_score = float(score_text.split("/")[0])
                        else:
                            for word in score_text.split():
                                if word.replace('.', '').isdigit():
                                    improvement_score = float(word)
                                    break
                    except:
                        pass
                
                if line.startswith("- ") and any(x in line.lower() for x in ["improv", "better", "strength"]):
                    key_improvements.append(line[2:])
                
                if line.startswith("- ") and any(x in line.lower() for x in ["focus", "need", "miss", "lack", "area"]):
                    areas_to_focus.append(line[2:])
            
            return {
                "improvement_score": improvement_score or 5.0,  # Default to 5 if parsing fails
                "feedback": feedback,
                "key_improvements": key_improvements,
                "areas_to_focus": areas_to_focus or ["Continue refining the analysis"]
            }
            
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            # Return default values if parsing fails
            return {
                "improvement_score": 5.0,
                "feedback": "Error parsing evaluation response.",
                "key_improvements": ["Parsing error - unable to extract specific improvements"],
                "areas_to_focus": ["Fix evaluation parsing", "Continue refining the analysis"]
            }
        
    def save_feedback(self, rfp_path: str, result, evaluation):
        """Save feedback and results for future reference"""
        rfp_name = os.path.basename(rfp_path)
        
        feedback_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rfp_path": rfp_name,
            "key_findings": [f"{r.category}: {r.risk}" for r in result.risk_clauses],
            "improvement_areas": evaluation["areas_to_focus"],
            "feedback_score": evaluation["improvement_score"],
            "risk_level": result.overall_risk_level
        }
        
        self.feedback_history.append(feedback_entry)
        self._save_feedback_history()

    def iterative_analysis(self, rfp_path: str, eligibility_result=None, compliance_result=None) -> Dict[str, Any]:
        """Run multiple iterations of analysis with feedback loop"""
        best_result = None
        best_score = 0
        previous_result = None
        
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n📝 Starting Analysis Iteration {iteration}/{self.MAX_ITERATIONS}...")
            
            # Run analysis
            current_result = self.analyze_risk_clauses(
                rfp_path,
                eligibility_result=eligibility_result,
                compliance_result=compliance_result,
                iteration=iteration,
                previous_result=previous_result
            )
            
            # Evaluate against previous result
            evaluation = self.evaluate_analysis(current_result, previous_result)
            
            # Save feedback
            self.save_feedback(rfp_path, current_result, evaluation)
            
            # Print evaluation
            print(f"\n📊 EVALUATION (Iteration {iteration}):")
            print(f"Improvement score: {evaluation['improvement_score']}")
            print("Key improvements:")
            for imp in evaluation.get("key_improvements", []):
                print(f"- {imp}")
            print("\nAreas to focus on:")
            for area in evaluation.get("areas_to_focus", []):
                print(f"- {area}")
            
            # Update best result if this is better
            current_score = evaluation["improvement_score"]
            if current_score is not None and current_score > best_score:
                best_score = current_score
                best_result = current_result
            elif best_result is None:
                best_result = current_result
            
            # Set up for next iteration
            previous_result = current_result
            
            # If we've reached a very good score, we can exit early
            if current_score is not None and current_score >= 9:
                print(f"\n✨ Achieved excellent score ({current_score}/10). Ending iterations early.")
                break
        
        print("\n🏆 FINAL BEST ANALYSIS:")
        print(json.dumps(best_result.dict(), indent=2))
        return best_result

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
    agent = ReActRiskClauseAnalyzerAgent()

    agent.clear_vector_store()

    eligibility_result = None
    compliance_result = None

    try:
        if os.path.exists(config.ELIGIBILITY_JSON):
            with open(config.ELIGIBILITY_JSON, "r") as f:
                eligibility_result = json.load(f)

        if os.path.exists(config.COMPLIANCE_CHECKLIST_JSON):
            with open(config.COMPLIANCE_CHECKLIST_JSON, "r") as f:
                compliance_result = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load previous agent outputs: {e}")

    print("\n🔎 Launching ReAct Risk Clause Analyzer...")
    result = agent.iterative_analysis(
        "./documents/RFPs/ELIGIBLE RFP - 2.pdf",
        eligibility_result=eligibility_result,
        compliance_result=compliance_result
    )

    print("\n📊 SAVING FINAL RISK ANALYSIS OUTPUT:")
    with open(config.CONTRACT_RISK_JSON, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
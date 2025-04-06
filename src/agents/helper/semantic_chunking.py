from typing import List, Dict
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


class SemanticChunker:
    """
    A semantic chunking utility that creates more contextually coherent document chunks
    compared to traditional character-based chunking.
    """
    def __init__(self, llm):
        self.llm = llm
        self.chunk_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                You are a document chunking specialist. Your goal is to break the provided text into 
                semantically coherent chunks that maintain the context of requirements, criteria, or 
                technical specifications. Focus on preserving:
                
                1. Complete requirements sections
                2. Related criteria that should stay together
                3. Technical specifications that belong together
                4. Logical section boundaries
                
                Create chunks of approximately 800-1000 characters that maintain semantic coherence.
            """),
            HumanMessage(content="Text to chunk semantically:\n\n{text}")
        ])
        self.chunking_chain = LLMChain(llm=self.llm, prompt=self.chunk_prompt)
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into semantic chunks preserving context"""
        # First use a basic splitter to break very large text into manageable pieces
        initial_splitter = RecursiveCharacterTextSplitter(
            separator="\n\n",
            chunk_size=4000,
            chunk_overlap=200
        )
        initial_chunks = initial_splitter.split_text(text)
        
        result_chunks = []
        
        # Process each initial chunk with the LLM for semantic chunking
        for i, chunk in enumerate(initial_chunks):
            try:
                # Get semantic chunks recommendation from LLM
                response = self.chunking_chain.invoke({"text": chunk})
                semantic_chunks_text = response['text']
                
                # Split by semantic chunk markers if present, otherwise by paragraphs
                if "CHUNK" in semantic_chunks_text:
                    semantic_chunks = semantic_chunks_text.split("CHUNK")[1:]
                else:
                    # Fallback to paragraph splitting
                    semantic_chunks = semantic_chunks_text.split("\n\n")
                
                # Create document chunks with metadata
                for j, semantic_chunk in enumerate(semantic_chunks):
                    if len(semantic_chunk.strip()) > 0:
                        chunk_metadata = metadata.copy() if metadata else {}
                        chunk_metadata.update({
                            "semantic_chunk_index": f"{i}-{j}",
                            "semantic_group": i
                        })
                        result_chunks.append({
                            "content": semantic_chunk.strip(),
                            "metadata": chunk_metadata
                        })
            except Exception as e:
                print(f"Error in semantic chunking: {e}")
                # Fallback to simple paragraph chunking
                fallback_chunks = chunk.split("\n\n")
                for j, fallback_chunk in enumerate(fallback_chunks):
                    if len(fallback_chunk.strip()) > 0:
                        chunk_metadata = metadata.copy() if metadata else {}
                        chunk_metadata.update({
                            "semantic_chunk_index": f"{i}-{j}-fallback",
                            "semantic_group": i
                        })
                        result_chunks.append({
                            "content": fallback_chunk.strip(),
                            "metadata": chunk_metadata
                        })
        
        return result_chunks
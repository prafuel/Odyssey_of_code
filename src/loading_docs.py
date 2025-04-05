import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain.embeddings import HuggingFaceEmbeddings

# Map file extensions to appropriate loaders
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader
}

def load_document(file_path, doc_type):
    """
    Load a single document file based on its extension
    
    Args:
        file_path (str): Path to the document file
        doc_type (str): Type of document (e.g., "company_document", "rfp_document")
        
    Returns:
        list: List of document chunks with metadata
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
        
    file_extension = os.path.splitext(file_path)[1].lower()
    
    
    if file_extension not in LOADER_MAPPING:
        print(f"Error: Unsupported file format for {file_path}")
        return []
    
    try:
        # Create appropriate loader based on file extension
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf8")
        else:
            loader = LOADER_MAPPING[file_extension](file_path)
            
        # Load the document
        documents = loader.load()
        
        # Add metadata to each document page/chunk
        for doc in documents:
            doc.metadata["document_type"] = doc_type
            doc.metadata["source_file"] = file_path
            
        print(f"Successfully loaded: {file_path}")
        return documents
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def split_documents_semantic(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks based on semantic meaning
    
    Args:
        documents (list): List of document objects
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    if not documents:
        return []
    
    # First use RecursiveCharacterTextSplitter to get initial chunks
    # This helps break very large documents into manageable pieces
    initial_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger initial chunks
        chunk_overlap=300,
        length_function=len,
    )
    
    initial_chunks = initial_splitter.split_documents(documents)
    print(f"Created {len(initial_chunks)} initial chunks")
    
    # Initialize embeddings for semantic chunking
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Use semantic chunker for content-aware splitting
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # Higher threshold means fewer breaks
    )
    
    # Apply semantic chunking to each initial chunk
    semantic_chunks = []
    for chunk in initial_chunks:
        try:
            semantic_chunks.extend(semantic_splitter.split_documents([chunk]))
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            # Fall back to character splitting if semantic chunking fails
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            semantic_chunks.extend(fallback_splitter.split_documents([chunk]))
    
    print(f"Created {len(semantic_chunks)} semantic chunks")
    
    return semantic_chunks
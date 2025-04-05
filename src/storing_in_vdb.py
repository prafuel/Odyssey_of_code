from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store(documents, persist_directory="./chroma_db"):
    """
    Create and persist a vector store from documents
    
    Args:
        documents (list): List of document chunks
        persist_directory (str): Directory to store the vector database
        
    Returns:
        Chroma: Vector store object
    """
    if not documents:
        print("No documents provided to store.")
        return None
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and persist vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Documents have been processed and stored in the vector database at {persist_directory}")
    print(f"Total chunks in vector DB: {vectordb._collection.count()}")
    
    return vectordb

def load_vector_store(persist_directory="./chroma_db"):
    """
    Load an existing vector store
    
    Args:
        persist_directory (str): Directory where the vector database is stored
        
    Returns:
        Chroma: Vector store object
    """
    try:
        # Initialize embeddings (must match what was used to create the DB)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load the persisted database
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        print(f"Loaded vector database from {persist_directory}")
        print(f"Total chunks in vector DB: {vectordb._collection.count()}")
        
        return vectordb
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

def query_database(vectordb, query_text, k=3):
    """
    Query the vector database
    
    Args:
        vectordb: Vector database object
        query_text (str): Query text
        k (int): Number of results to return
        
    Returns:
        list: List of relevant document chunks
    """
    if not vectordb:
        print("No vector database available.")
        return []
        
    docs = vectordb.similarity_search(query_text, k=k)
    print(f"\nQuery: {query_text}")
    print(f"Top {k} relevant document chunks:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}:")
        print(f"Source: {doc.metadata.get('source_file')}")
        print(f"Type: {doc.metadata.get('document_type')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    return docs
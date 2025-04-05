import os
import sys
from src.loading_docs import load_document, split_documents_semantic
from src.storing_in_vdb import create_vector_store, query_database, load_vector_store

def process_documents(company_doc_path, rfp_doc_path, persist_dir="./chroma_db"):
    """
    Process company and RFP documents and store them in a vector database
    
    Args:
        company_doc_path (str): Path to company document
        rfp_doc_path (str): Path to RFP document
        persist_dir (str): Directory to store the vector database
        
    Returns:
        object: Vector database object
    """
    # check if database is exist or not
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Found existing vector database at {persist_dir}")
        vectordb = load_vector_store(persist_dir)
        if vectordb:
            print("Using existing database")
            return vectordb

    # Load documents
    print(f"Loading company document: {company_doc_path}")
    company_doc = load_document(company_doc_path, "company_document")
    print(f"Loaded {len(company_doc)} pages/sections from company document")
    
    print(f"Loading RFP document: {rfp_doc_path}")
    rfp_doc = load_document(rfp_doc_path, "rfp_document")
    print(f"Loaded {len(rfp_doc)} pages/sections from RFP document")
    
    # Combine all documents
    all_docs = company_doc + rfp_doc
    
    if len(all_docs) == 0:
        print("No document content loaded. Please check your file paths and formats.")
        return None
    
    # Split documents into chunks
    chunks = split_documents_semantic(all_docs)
    
    # Create and store vector database
    vectordb = create_vector_store(chunks, persist_dir)
    
    return vectordb

def main():
    # Get paths from command line arguments or use defaults
    company_doc_path = "./documents/company_data/Company Data.docx"
    rfp_doc_path = "./documents/RFPs/ELIGIBLE RFP - 1.pdf"
    
    # Process documents
    company_vd = process_documents(company_doc_path, rfp_doc_path, "./chroma/company_vd")
    rfs_vd = process_documents(company_doc_path, rfp_doc_path, "./chroma/rfp_vd")
    
    # Example queries
    if company_vd:
        query_database(company_vd, "What are the key requirements in the RFP?")
        query_database(company_vd, "What are the main services offered by the company?")
    
    if rfs_vd:
        query_database(rfs_vd, "What are the key requirements in the RFP?")
        query_database(rfs_vd, "What are the main services offered by the company?")

if __name__ == "__main__":
    main()
# app/embeddings.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # 500 characters per chunk
        chunk_overlap=100      # 50 char overlap between chunks
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from PDF")
    return chunks

# app/embeddings.py (add to existing file)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # Use free HuggingFace embeddings (no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save locally so you don't rebuild every time
    vector_store.save_local("faiss_index")
    print("Vector store created and saved!")
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local("faiss_index", embeddings,
                             allow_dangerous_deserialization=True)
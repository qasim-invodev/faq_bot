import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def load_pdf_chunks(pdf_path):
    """Load a PDF and split it into text chunks."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

def create_or_load_vectorstore(chunks, index_path="faiss_index"):
    """Embed chunks and load/create a FAISS vector store."""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_path):
        print("🔄 Loading existing vector store...")
        return FAISS.load_local(index_path, embeddings)
    
    print("📦 Creating new vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore
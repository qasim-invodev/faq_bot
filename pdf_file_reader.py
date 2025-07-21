from langchain_community.document_loaders import PyMuPDFLoader  # LangChain loader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def pdf_file_reader(file_path):
    # Load the PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print("Documents PDF loaded", documents)

    # Split into chunks for embeddings
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return chunks


def make_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()

    # Create the FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Save it to disk
    db.save_local("faiss_index_pdf")
    return embeddings

def load_vector_store(embeddings):
    # Load PDF vector store
    db = FAISS.load_local("faiss_index_pdf", embeddings)

    # Build Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=db.as_retriever()
    )
    return qa

def load_pdf_file_langchain(file_path):
    chunks = pdf_file_reader(file_path)
    embeddings = make_faiss_index(chunks)
    qa = load_vector_store(embeddings)
    return qa
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def text_file_reader(file_path):
    # Load and split the text
    loader = TextLoader(file_path)
    documents = loader.load()
    print("Documents TEXT loaded", documents)
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return chunks

def make_faiss_index(chunks):
    # Create vector store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Optional: Save the index
    db.save_local("faiss_index")
    return embeddings

def load_vector_store(embeddings):
    # Load the saved vector store
    db = FAISS.load_local("faiss_index", embeddings)

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=db.as_retriever()
    )
    return qa

def load_text_file_langchain(file_path):
    chunks = text_file_reader(file_path)
    embeddings = make_faiss_index(chunks)
    qa = load_vector_store(embeddings)
    return qa
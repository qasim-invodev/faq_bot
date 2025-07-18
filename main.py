from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load and split the text
loader = TextLoader("faqs.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Optional: Save the index
db.save_local("faiss_index")

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# Load the saved vector store
db = FAISS.load_local("faiss_index", OpenAIEmbeddings())

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=db.as_retriever()
)

# Ask a question
while True:
    question = input("Ask a question (or 'exit'): ")
    if question.lower() == 'exit':
        break
    result = qa.run(question)
    print("Answer:", result)
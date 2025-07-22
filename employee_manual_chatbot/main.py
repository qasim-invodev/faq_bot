import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from utils import load_pdf_chunks, create_or_load_vectorstore

# ✅ Load env vars
load_dotenv()
pdf_path = "sample.pdf"

# 📄 Load and process PDF
chunks = load_pdf_chunks(pdf_path)

# 🔎 Load or create FAISS vector store
vectorstore = create_or_load_vectorstore(chunks)

# 🧠 Create conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 💬 LLM + Retrieval QA Chain with memory
llm = ChatOpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# 🧪 Interactive chatbot loop
print("🤖 Ask questions about the PDF (type 'exit' to quit):")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.run(query)
    print("\nBot:", result)
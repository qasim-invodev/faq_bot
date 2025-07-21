from text_file_reader import load_text_file_langchain
from pdf_file_reader import load_pdf_file_langchain

# Get file path from user
file_path = input("Enter the path to your file (PDF or text): ")

# Determine file type and load appropriate QA system
if file_path.lower().endswith('.pdf'):
    print("Loading PDF file...")
    qa = load_pdf_file_langchain(file_path)
elif file_path.lower().endswith('.txt'):
    print("Loading text file...")
    qa = load_text_file_langchain(file_path)
else:
    print("Unsupported file type. Please use PDF or text files.")
    exit()

# Ask a question
while True:
    question = input("Ask a question (or 'exit'): ")
    if question.lower() == 'exit':
        break
    result = qa.run(question)
    print("Answer:", result)
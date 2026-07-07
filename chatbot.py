import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()  

loader = PyPDFLoader('test.pdf')
pdf_loader = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pdf_loader)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding=embeddings)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_chatbot(query):
    result = db.similarity_search(query)
    context = "\n\n".join(doc.page_content for doc in result)
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """
    response = llm.invoke(prompt)
    print(response.content)


ask_chatbot("Where is the store located")
ask_chatbot("What is the usp of the company")
ask_chatbot("What are some products available in the store")
ask_chatbot("Are there any vacancy available in the company")
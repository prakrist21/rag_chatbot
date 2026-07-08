from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs(folder_path="docs"):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load() 
            all_docs.extend(pages)
            print(f"Loaded {filename}: {len(pages)} pages")
    return all_docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(chunks, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_rag",
        persist_directory=persist_directory
    )
    print("Vector store built and persisted.")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs("docs")
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
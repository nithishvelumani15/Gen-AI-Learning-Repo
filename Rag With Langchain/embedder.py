import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import load_documents
from document_chunker import create_chunks

load_dotenv()

CHROMA_PATH = "./chroma_langchain_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

FOLDER = r"C:\Users\nithi\Desktop\My Projects\Learning\RAG Tuturial\Gen-AI-Learning-Repo\Rag With Langchain\HR_Policy_Documents"
def build_index():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Existing index deleted")
    docs = load_documents(FOLDER)
    chunks = create_chunks(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("success)
if __name__ == "__main__":
    build_index()

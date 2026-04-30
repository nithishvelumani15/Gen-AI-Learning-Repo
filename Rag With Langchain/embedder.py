# indexer.py
# Run this script ONLY when:
# - New documents are added to the folder
# - Existing documents are updated
# - First time setup
# DO NOT run this every time you want to chat

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
    # Step 1 — Wipe existing ChromaDB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Existing index wiped")

    # Step 2 — Load documents
    docs = load_documents(FOLDER)

    # Step 3 — Chunk documents
    chunks = create_chunks(docs)

    # Step 4 — Embed and store
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Index built successfully: {vectorstore._collection.count()} chunks stored")
    print(f"ChromaDB saved at: {CHROMA_PATH}")

if __name__ == "__main__":
    build_index()
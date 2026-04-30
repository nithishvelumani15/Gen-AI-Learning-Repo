from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from documentRetriver import file_content
from document_chunking import chunk_text
from datetime import datetime

load_dotenv()

client = chromadb.PersistentClient(path="./RAG Tuturial/chroma_db")
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client.delete_collection("policies")
collection = client.create_collection("policies")
policies = file_content()
for pol in policies:
    if not pol["text"].strip():
        continue
    doc_text = chunk_text(pol["text"])
    source_name = pol["Source"].replace(" ", "_").replace(".", "_")
    policy_embeddings = embed_model.encode(doc_text).tolist()
    ids = [f"policy_{source_name}_{i}" for i in range(len(doc_text))]
    collection.add(
        documents=doc_text,
        embeddings=policy_embeddings,
        ids=ids
    )
print("success")
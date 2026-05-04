from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
def create_chunks(documents: list[Document], chunk_size=400, chunk_overlap=50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    from document_loader import load_documents
    FOLDER = r"C:\Users\nithi\Desktop\My Projects\Learning\RAG Tuturial\Gen-AI-Learning-Repo\Rag With Langchain\HR_Policy_Documents"
    docs = load_documents(FOLDER)
    chunks = create_chunks(docs)

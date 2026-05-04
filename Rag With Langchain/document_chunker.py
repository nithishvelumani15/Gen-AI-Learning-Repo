from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents: list[Document], chunk_size=400, chunk_overlap=50) -> list[Document]:
    # RecursiveCharacterTextSplitter splits on paragraphs first,
    # then sentences, then words — smarter than plain word splitting
    # Also preserves metadata — each chunk keeps source + type from parent doc
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk: {chunks[0].page_content[:150]}")
    print(f"Sample metadata: {chunks[0].metadata}")
    return chunks


if __name__ == "__main__":
    from document_loader import load_documents
    FOLDER = r"C:\Users\nithi\Desktop\My Projects\Learning\RAG Tuturial\Gen-AI-Learning-Repo\Rag With Langchain\HR_Policy_Documents"
    docs = load_documents(FOLDER)
    chunks = create_chunks(docs)
    print(f"\nTotal chunks: {len(chunks)}")

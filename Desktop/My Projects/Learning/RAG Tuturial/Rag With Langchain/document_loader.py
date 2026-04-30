# document_loader.py
# Responsibility: Load files from folder, return LangChain Document objects

from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from markitdown import MarkItDown
from langchain_core.documents import Document

md_converter = MarkItDown()

def get_doc_content(file: Path) -> Document | None:
    try:
        result = md_converter.convert(str(file))
        doc_text = result.text_content

        if not doc_text.strip():
            print(f"{file.name} skipped: empty content")
            return None

        # Only change from your original — return Document instead of dict
        return Document(
            page_content=doc_text,
            metadata={
                "source": file.name,
                "type": file.suffix
            }
        )
    except Exception as e:
        print(f"{file.name} skipped: {e}")
        return None

def load_documents(folder_path: str) -> list[Document]:
    print(f"Document loading start: {datetime.now()}")

    all_files = [f for f in Path(folder_path).iterdir() if f.is_file()]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_doc_content, all_files))

    documents = [doc for doc in results if doc is not None]

    print(f"Document loading end: {datetime.now()}")
    print(f"Successfully loaded: {len(documents)} documents")
    return documents


if __name__ == "__main__":
    FOLDER = r"C:\Users\nithi\Desktop\My Projects\Learning\Langchain\Rag With Langchain\HR_Policy_Documents"
    docs = load_documents(FOLDER)
    for doc in docs:
        print(f"{doc.metadata['source']}: {len(doc.page_content)} chars")
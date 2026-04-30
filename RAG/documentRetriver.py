from pathlib import Path
import pdfplumber
from docx import Document
from markitdown import MarkItDown
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
md_converter = MarkItDown()
def file_content():
    print(f"Document Retrievel start time: {datetime.now()}")
    all_files = []
    folder_path = Path(
        r"C:\Users\nithi\Desktop\My Projects\Learning\PythonLearning1\RAG\HR_Policy_Documents"
    )
    for file in folder_path.iterdir():
        if file.is_file():
            all_files.append(file)
    with ThreadPoolExecutor() as executor:
        result  = list(executor.map(get_doc_content,all_files))
    print(f"Document Retrievel end time: {datetime.now()}")
    return result
def get_doc_content(file):
    try:
        doc_text = ""
        result = md_converter.convert(str(file))
        doc_text = result.text_content
        '''
        if file.suffix == ".txt":
            with open(file, 'r') as f:
                doc_text = f.read()
        elif file.suffix == ".pdf":
            doc_text=""
            with pdfplumber.open(file) as f:
                for page in f.pages:
                    doc_text += (page.extract_text() or "")+"\n"
        elif file.suffix ==".docx":
            doc_text=""
            doc = Document(file)
            for p in doc.paragraphs:
                if p.text.strip():
                    doc_text+=p.text + "\n"
        '''
    except Exception as e:
        print(f"{file.name} skipped {e}")

    print(f"file executed: {file.name}")
    return{
            "text": doc_text,
            "Source": file.name,
            "Type": file.suffix
        }

if __name__ == "__main__":
    file_content()




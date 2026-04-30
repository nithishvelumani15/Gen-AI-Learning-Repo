def chunk_text(doc_text, chunk_size=400, overlapping=50):
    doc_words = doc_text.split()
    chunks = []
    start = 0
    while(start<len(doc_words)):
        end = start+chunk_size
        chunk = " ".join(doc_words[start:end])
        chunks.append(chunk)
        start += chunk_size-overlapping
    return chunks
if __name__ == "__main__":
    chunk_text()

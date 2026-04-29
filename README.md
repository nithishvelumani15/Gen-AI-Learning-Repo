# RAG HR Assistant

A small retrieval-augmented generation project for HR policy questions.

It reads HR policy documents, chunks and embeds them into ChromaDB, then answers user queries using retrieved policy context and an LLM backend.

## Files

- `documentRetriver.py`: loads HR policy documents
- `document_chunking.py`: splits text into chunks
- `embedding.py`: builds vector embeddings and stores them in ChromaDB
- `ollamaAgent.py`: interactive query loop using Ollama
- `agent.py`: similar assistant using Google Gemini

## Usage

1. Run `embedding.py` to build the vector store.
2. Run `ollamaAgent.py` to ask questions.

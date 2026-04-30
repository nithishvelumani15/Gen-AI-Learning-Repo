import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter


load_dotenv()

CHROMA_PATH = "./chroma_langchain_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def build_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in docs
        )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a literal document-reading HR assistant. 
                    Answer using the context below and the conversation history.
                    If the answer is not in the context, you may use conversation history.
                    you should maintain conversation like a real HR ,let's say greeting and other human behaviours.
                    If the answer is not available, say: 'Please reach out to your respective HR.'
                    Provide plain text only, no markdown.
                    Context: {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    rag_chain = (
        {
        "context": itemgetter("input") | retriever | format_docs,  
        "input": itemgetter("input"),                              
        "history": itemgetter("history"),                           
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    session_store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,          
        input_messages_key="input",   
        history_messages_key="history" 
    )


    return chain_with_memory

if __name__ == "__main__":
    rag_chain = build_rag_chain()

    while True:
        print("\nEnter your question. Type 'quit' or 'exit' to stop.")
        user_query = input("Question: ")
        user_query_passing = {"input": user_query}
        config = {"configurable": {"session_id": "nithish_session_1"}}
        if user_query.lower() in ['exit', 'quit'] or user_query == "":
            print("Stopping the session.")
            break

        answer = rag_chain.invoke(user_query_passing, config=config)
        print(f"\nHR Assistant: {answer}")
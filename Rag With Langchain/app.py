import streamlit as st
import os
from dotenv import load_dotenv
from rag_chain import build_rag_chain

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="HR Assistant Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for the RAG chain and Chat History
if "rag_chain" not in st.session_state:
    with st.spinner("Initializing AI Engine..."):
        st.session_state.rag_chain = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Settings
with st.sidebar:
    st.header("⚙️ Settings")
    session_id = st.text_input("Session ID:", value="streamlit_session_1")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 HR Assistant Chatbot")
st.caption("Ask me about HR policies, leave, and benefits.")

# --- DISPLAY CHAT HISTORY ---
# This ensures previous messages stay visible on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT LOGIC ---
# st.chat_input is much more reliable than st.text_input + st.button
if prompt := st.chat_input("How can I help you today?"):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                user_query_input = {"input": prompt}
                config = {"configurable": {"session_id": session_id}}
                
                response = st.session_state.rag_chain.invoke(
                    user_query_input, 
                    config=config
                )
                
                st.markdown(response)
                
                # 4. Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
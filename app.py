# app.py

import streamlit as st
import os
from dotenv import load_dotenv

# Import our RAG processor
from rag_processor import create_vector_store, get_conversation_chain

# Load environment variables from .env file
load_dotenv()

# --- Main Application Logic ---
def main():
    # Set up the title and a brief description
    st.set_page_config(page_title="CogniQuery", layout="wide")
    st.title("🧠 CogniQuery: Your Personal Document Intelligence Engine")
    st.write("Upload a PDF document and ask any question. CogniQuery will read the document and provide precise, context-aware answers.")

    # Initialize session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # --- UI Components ---
    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.sidebar.button("Process Document"):
            with st.spinner("Processing document... This may take a while for large files."):
                # 1. Save the uploaded file temporarily
                docs_dir = "./uploaded_docs"
                if not os.path.exists(docs_dir):
                    os.makedirs(docs_dir)
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. Create the vector store
                create_vector_store(file_path)
                
                # 3. Create the conversation chain and store it in session state
                st.session_state.conversation = get_conversation_chain()
                st.sidebar.success("Document processed successfully! You can now ask questions.")

    question = st.text_input("Ask a question about your document:")
    if question:
        if st.session_state.conversation:
            with st.spinner("Finding the answer..."):
                response = st.session_state.conversation({'question': question})
                st.session_state.chat_history = response['chat_history']
                
                st.subheader("Answer:")
                st.write(response['answer'])

        else:
            st.warning("Please upload and process a document first.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built by Mohammed Zaid Ahmed")
    st.sidebar.markdown("Powered by Streamlit, LangChain, and Groq")

if __name__ == '__main__':
    main()
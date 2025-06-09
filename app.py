# app.py

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the title and a brief description
st.set_page_config(page_title="CogniQuery", layout="wide")
st.title("🧠 CogniQuery: Your Personal Document Intelligence Engine")
st.write("Upload a PDF document and ask any question. CogniQuery will read the document and provide precise, context-aware answers.")

# UI Components
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question about the document", "")

if st.sidebar.button("Process Document and Ask"):
    if uploaded_file is not None and question:
        with st.spinner('Processing document and finding answers... This may take a moment.'):
            # Save the uploaded file temporarily
            docs_dir = "./uploaded_docs"
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Placeholder for the real processing
            st.subheader("Answer:")
            st.write(f"This is a placeholder answer for your question: '{question}'. The real AI-generated answer will appear here, powered by Groq!")
            
    elif not uploaded_file:
        st.sidebar.warning("Please upload a PDF document first.")
    elif not question:
        st.warning("Please enter a question.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Mohammed Zaid Ahmed")
st.sidebar.markdown("Powered by Streamlit, LangChain, and Groq")
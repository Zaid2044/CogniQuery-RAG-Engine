# rag_processor.py

import os
from dotenv import load_dotenv

# Document Loading
from langchain_community.document_loaders import PyMuPDFLoader

# Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store and Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM and Chat/Prompt templates
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Load API Key ---
load_dotenv()

# --- Global Variables ---
VECTOR_STORE_PATH = "vectorstore/db_faiss"

def create_vector_store(file_path):
    """
    Loads a PDF, splits it into chunks, creates embeddings, and stores them in a FAISS vector store.
    
    Args:
        file_path (str): The path to the PDF file.
    """
    print("Creating vector store...")
    # 1. Load the document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the document.")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split document into {len(docs)} chunks.")

    # 3. Create embeddings model
    # We will use a powerful, open-source model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create vector store and save it locally
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created and saved at {VECTOR_STORE_PATH}")
    return db

def get_conversation_chain():
    """
    Creates and returns a conversational retrieval chain.
    """
    print("Loading vector store and creating conversation chain...")
    # 1. Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Load the vector store from local storage
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # 3. Create a retriever interface from the vector store
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

    # 4. Set up the LLM with Groq
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

    # 5. Define the prompt template
    # This guides the LLM to answer based ONLY on the provided context.
    template = """
    You are an intelligent assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise and relevant.
    
    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 6. Set up memory to allow for conversational context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 7. Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    print("Conversation chain created successfully.")
    return conversation_chain
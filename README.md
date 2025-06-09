# 🧠 CogniQuery: Intelligent Document Query Engine

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-purple.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference%20by-Groq-green.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-orange.svg)](https://streamlit.io/)

An advanced, conversational AI engine that allows you to "chat" with your documents. Built with a robust RAG (Retrieval-Augmented Generation) pipeline, CogniQuery provides fast, accurate, and context-aware answers from your PDFs, complete with source attribution to eliminate hallucinations.

---

## 🚀 The Problem

Enterprises, legal teams, and researchers are drowning in data locked away in unstructured documents like PDFs and reports. Traditional keyword search is inefficient, time-consuming, and often fails to capture the semantic context of a query. This knowledge retrieval bottleneck costs time, money, and delays critical decision-making.

## 💡 Our Solution: The RAG Architecture

CogniQuery solves this by implementing a state-of-the-art **Retrieval-Augmented Generation (RAG)** pipeline. This approach enhances Large Language Models (LLMs) by grounding them in factual knowledge from your documents, ensuring responses are accurate and trustworthy.

Here’s how it works:
1.  **Ingestion & Indexing**: When a PDF is uploaded, it's broken down into smaller, meaningful text chunks.
2.  **Vector Embeddings**: Each chunk is converted into a numerical representation (a vector) using the powerful `all-MiniLM-L6-v2` sentence-transformer model.
3.  **Vector Store**: These vectors are stored in a highly efficient `FAISS` vector database, creating a searchable knowledge index.
4.  **Retrieval & Generation**:
    *   When you ask a question, it's also converted into a vector.
    *   The FAISS database is searched to find the most semantically similar text chunks from the document.
    *   These relevant chunks, along with your original question, are fed to a cutting-edge LLM (`Llama-3` via the lightning-fast **Groq API**).
    *   The LLM generates a precise, human-like answer based **only** on the provided context.

This architecture ensures answers are not just fluent but are also factually grounded in the source material, effectively eliminating the "hallucination" problem common in standalone LLMs.

## 🛠️ Technology Stack

*   **Backend & Logic:** Python, LangChain
*   **Web Framework:** Streamlit
*   **LLM Inference:** Groq API (Llama-3 8B)
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
*   **Vector Database:** `FAISS` (Facebook AI Similarity Search)
*   **PDF Processing:** `PyMuPDF`

## 🏁 Getting Started

### Prerequisites

*   Python 3.9+
*   Git
*   A [Groq API Key](https://console.groq.com/keys)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Zaid2044/CogniQuery-RAG-Engine.git
    cd CogniQuery-RAG-Engine
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate
    
    # On MacOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    *   Create a file named `.env` in the root of the project directory.
    *   Add your Groq API key to the `.env` file:
      ```
      GROQ_API_KEY="gsk_YourActualKeyFromGroq"
      ```

### Running the Application

Once the installation is complete, run the following command from your terminal:

```bash
streamlit run app.py
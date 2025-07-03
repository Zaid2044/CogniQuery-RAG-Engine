# 🧠 CogniQuery: Intelligent Document Query Engine

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-purple.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference%20by-Groq-green.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-orange.svg)](https://streamlit.io/)

**Repository:** [CogniQuery-RAG-Engine](https://github.com/Zaid2044/CogniQuery-RAG-Engine)  
**Author:** MOHAMMED ZAID AHMED

CogniQuery is a conversational AI engine that lets you **chat with your PDFs**. Powered by Retrieval-Augmented Generation (RAG), it gives accurate, source-grounded responses to your questions — with blazing fast inference via **Groq** and clean UX via **Streamlit**.

---

## 🚨 The Problem

Professionals and organizations deal with massive amounts of unstructured documents. Traditional keyword-based search is slow, context-blind, and ineffective when you need precise answers.

---

## 💡 The Solution: RAG Architecture

CogniQuery uses a **Retrieval-Augmented Generation pipeline** to connect language models with your document knowledge base.

### 🧠 How It Works

1. **PDF Ingestion**
   → Breaks down PDFs into structured, chunked text
2. **Vector Embedding**
   → Converts chunks into semantic vectors with `all-MiniLM-L6-v2`
3. **FAISS Indexing**
   → Stores embeddings in a fast, searchable vector store
4. **Query & Answer**
   → Queries are vectorized → relevant chunks retrieved
   → Combined with your query and sent to **LLaMA 3 (8B)** via **Groq**
   → Outputs natural, context-aware, and source-grounded responses

✅ **No hallucinations**
✅ **Fully local document context**
✅ **Near-instant responses**

---

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **LLM Inference:** Groq (LLaMA 3 8B)
* **RAG Framework:** LangChain
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector DB:** FAISS
* **PDF Parsing:** PyMuPDF

---

## ⚙️ Getting Started

### 📦 Prerequisites

* Python 3.9+
* Git
* [Groq API Key](https://console.groq.com/keys)

### 🔧 Installation

```bash
git clone https://github.com/Zaid2044/CogniQuery-RAG-Engine.git
cd CogniQuery-RAG-Engine
```

#### Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure API Key

Create a `.env` file in the root:

```env
GROQ_API_KEY="gsk_your_actual_groq_key"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

> Upload a PDF → Ask any question → Get contextual answers with source references.

---

## 🧩 Future Upgrades

* Multi-file PDF support
* Answer highlighting in source document
* Support for image-based PDFs (OCR integration)
* Export QA history

---

## 📜 License

This project is licensed under the MIT License.

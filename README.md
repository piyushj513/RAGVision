# 🔍 RAGVision

**AI-powered assistant that answers questions based on uploaded PDFs, images, or plain text — combining OCR, semantic search, and generative AI.**

---

## ✨ Features

- 🧠 **Retrieval-Augmented Generation (RAG):** Combines file-based search with LLMs to provide accurate, document-aware answers.
- 📄 **Multi-format Support:** Upload PDFs, images (JPG/PNG), or plain text for context-based Q&A.
- 🔎 **OCR-Powered:** Extracts text from images using Tesseract for intelligent analysis.
- 🤖 **LLM Integration:** Uses Azure OpenAI for natural, fluent responses with HuggingFace embeddings for context matching.
- ⚡ **Real-time Streaming:** Interactive chat with streaming responses via FastAPI and Streamlit.
- 🗂️ **Session Memory:** Maintains context across user sessions for a coherent conversation experience.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, LlamaIndex, Azure OpenAI, HuggingFace, Tesseract OCR
- **Frontend:** Streamlit (chat-style interface)
- **PDF/Text Extraction:** PyPDF2, Tesseract

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
https://github.com/piyushj513/RAGVision.git
cd RAGVision
```

### 2. Install deps and add .env keys
```bash
pip install -r requirements.txt
```
### 3. Run API

```bash
cd ./api
fastapi run ./chat.py
```

### 4. Run Frontend

```bash
cd ./frontend
streamlit run ./main.py
```
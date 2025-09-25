# Python AI Assistant 🤖

A full-stack AI-powered chat assistant built with:

- **FastAPI (Python)** → Backend API for embeddings, semantic search, and GPT integration  
- **FAISS** → Vector database to store and retrieve text efficiently  
- **React (JavaScript)** → Frontend UI for chat interface  
- **OpenAI API** → Used for embeddings & GPT responses  

---

## 🚀 Features
- Upload documents and store them as embeddings in FAISS  
- Ask questions → system finds the most relevant context  
- GPT generates an answer using **retrieved context + question**  
- Simple chat UI built with React  

---

## 🛠️ Tech Stack
- **Backend:** FastAPI, Pydantic, FAISS, NumPy, OpenAI SDK  
- **Frontend:** React, Axios (API calls), CSS (or Tailwind optional)  
- **Environment:** Python 3.9+ (via `pyenv`/`venv`), Node.js 18+  

---

## ⚡ Getting Started

### Backend (FastAPI + FAISS)
```bash
cd backend
python -m venv faiss_env
source faiss_env/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

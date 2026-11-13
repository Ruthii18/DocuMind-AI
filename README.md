# DocuMind – AI-Powered PDF Question Answering System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM%20%26%20Embeddings-Google%20Gemini-4285F4.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-FC521F.svg)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/Keyword%20Search-BM25Okapi-blue.svg)](https://github.com/dorianbrown/rank_bm25)

DocuMind is an enterprise-grade Retrieval-Augmented Generation (RAG) system that enables users to upload PDF documents and ask questions about their content using natural language. It combines **BM25 keyword search** and **ChromaDB vector semantic search** using **Reciprocal Rank Fusion (RRF)**, then passes grounded context to **Google Gemini** for fact-based answers with page citations.

---

## 🌟 Features

- **📄 Robust PDF Ingestion**: Extracts text across pages using `pypdf`, preserving original page numbers and skipping unreadable scanned documents.
- **✂️ Configurable Chunking**: Overlapping sliding-window chunking respecting word boundaries to preserve contextual meaning.
- **🔍 Hybrid Retrieval**:
  - **Semantic Search**: Vector similarity search in ChromaDB using Gemini text embeddings.
  - **Keyword Search**: BM25Okapi for precise lexical and keyword matching.
  - **Reciprocal Rank Fusion (RRF)**: Merges rankings from both semantic and keyword retrieval to yield superior context relevance.
- **🤖 Grounded Generation with Gemini**: Answers are constrained strictly to the provided context, preventing hallucinations and properly signaling when an answer is unavailable.
- **📍 Precise Source Citations**: Returns document filenames, page numbers, and snippet previews for transparent verifiability.
- **🖥️ Dual Interface**:
  - **Streamlit Web UI**: Interactive chat interface with document selector, upload progress, and collapsible source cards.
  - **FastAPI Backend**: RESTful API with automated Swagger UI (`/docs`).

---

## 🏗️ Architecture

```
PDF Upload ➔ Streamlit ➔ FastAPI ➔ PDF Extraction ➔ Chunking ➔ Embeddings ➔ ChromaDB
                                                                                  
User Question ➔ FastAPI ➔ [Vector Search + BM25 Search] ➔ RRF Fusion ➔ Context ➔ Gemini ➔ Answer + Citations
```

---

## 📁 Project Structure

```
DocuMind/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── config.py               # Environment & system configurations
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py        # PDF upload, listing & deletion endpoints
│   │   │   └── chat.py             # Q&A / Chat endpoint
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_service.py      # PDF text extraction & validation
│   │   │   ├── chunking_service.py # Text chunking with sliding window
│   │   │   ├── embedding_service.py# Google Gemini embeddings
│   │   │   ├── vector_store.py     # ChromaDB vector store wrapper
│   │   │   ├── retrieval_service.py# Hybrid search (Vector + BM25 + RRF)
│   │   │   └── rag_service.py      # Context assembly & LLM generation
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── document.py         # Pydantic schemas for documents
│   │       └── chat.py             # Pydantic schemas for chat & sources
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/
│   ├── app.py                      # Streamlit interactive user interface
│   └── requirements.txt
├── PROJECT REQUIREMENTS DOCUMENT.md
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- A Google Gemini API key (Get one from [Google AI Studio](https://aistudio.google.com/))

### 2. Setup Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your `GEMINI_API_KEY`:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   LLM_MODEL=gemini-1.5-flash
   EMBEDDING_MODEL=models/text-embedding-004
   CHUNK_SIZE=800
   CHUNK_OVERLAP=150
   TOP_K_RETRIEVAL=4
   RRF_K=60
   ```
5. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The backend will be available at `http://localhost:8000`.  
   Interactive API docs (Swagger): `http://localhost:8000/docs`.

### 3. Setup Frontend

1. In a new terminal window, navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Open your browser to `http://localhost:8501`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check backend health status |
| `POST` | `/documents/upload` | Upload and index a PDF document |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{document_id}` | Remove a document and its stored vectors |
| `POST` | `/chat/ask` | Ask a question about uploaded documents |

### Example: Upload Document (`POST /documents/upload`)
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@sample.pdf"
```

Response:
```json
{
  "message": "Document processed successfully",
  "document_id": "doc_a1b2c3d4",
  "filename": "sample.pdf",
  "total_pages": 12,
  "total_chunks": 38
}
```

### Example: Ask a Question (`POST /chat/ask`)
```bash
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc_a1b2c3d4", "question": "What is the primary objective of the system?"}'
```

Response:
```json
{
  "answer": "The primary objective is to enable natural language querying of PDF documents using Hybrid RAG.",
  "sources": [
    {
      "filename": "sample.pdf",
      "page": 2,
      "snippet": "DocuMind aims to address these problems by providing a document-based conversational interface..."
    }
  ]
}
```

---

## 📜 License
This project is developed under the MIT License.


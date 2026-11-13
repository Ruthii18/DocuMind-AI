# PROJECT REQUIREMENTS DOCUMENT
# DocuMind – AI-Powered PDF Question Answering System

**Project Type:** AI Engineering / Generative AI  
**Project Category:** Retrieval-Augmented Generation (RAG)  
**Version:** 1.0  
**Development Approach:** Backend-first  
**Frontend:** Streamlit  
**Backend:** FastAPI  

---

## 1. Project Overview

DocuMind is an AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their contents using natural language.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-grounded responses using a Large Language Model (LLM).

Instead of providing the entire document to the LLM for every question, DocuMind retrieves only the relevant text chunks and supplies them as context.

The system also supports hybrid retrieval by combining semantic search and keyword-based search to improve the relevance of retrieved information.

## 2. Problem Statement

Reading and searching lengthy PDF documents manually can be time-consuming.

Traditional keyword search may fail when users ask questions using different wording from the original document.

Additionally, general-purpose LLMs may generate answers that are not supported by an uploaded document.

DocuMind aims to address these problems by providing a document-based conversational interface that retrieves relevant information and generates answers grounded in the uploaded content.

## 3. Project Objectives

The primary objectives are:

1. Allow users to upload PDF documents.
2. Extract text from uploaded PDFs.
3. Divide extracted text into manageable chunks.
4. Generate embeddings using Gemini.
5. Store document chunks and embeddings in ChromaDB.
6. Implement semantic and keyword-based retrieval.
7. Combine retrieval results using Reciprocal Rank Fusion (RRF).
8. Generate answers using Gemini based on retrieved context.
9. Provide source references, including PDF page numbers.
10. Handle questions that cannot be answered using the available document context.

## 4. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core application development |
| Frontend | Streamlit | Interactive user interface |
| Backend | FastAPI | API development and request handling |
| LLM | Google Gemini | Answer generation |
| Embedding Model | Gemini Embedding | Text-to-vector conversion |
| PDF Processing | PyPDF | PDF text extraction |
| Vector Database | ChromaDB | Embedding storage and semantic retrieval |
| Keyword Retrieval | BM25 | Keyword-based document retrieval |
| Ranking | Reciprocal Rank Fusion | Combining retrieval rankings |
| Data Validation | Pydantic | Request and response validation |
| API Testing | Swagger UI | Backend endpoint testing |
| Version Control | Git and GitHub | Source code management |

## 5. System Architecture

The application consists of three major layers.

### A. Frontend Layer – Streamlit

Responsibilities:

- Provide a PDF upload interface.
- Allow users to enter questions.
- Display generated answers.
- Display source document references and page numbers.
- Show loading indicators and error messages.

### B. Backend Layer – FastAPI

Responsibilities:

- Receive PDF uploads.
- Extract PDF text.
- Process and chunk documents.
- Generate embeddings.
- Store document information.
- Handle user questions.
- Coordinate retrieval and answer generation.
- Return structured responses.

### C. AI and Retrieval Layer

Responsibilities:

- Generate document and query embeddings.
- Perform vector similarity search.
- Perform BM25 keyword retrieval.
- Combine rankings using RRF.
- Construct relevant context.
- Generate context-grounded answers using Gemini.

### Architecture Flow

PDF Upload → Streamlit → FastAPI → PDF Extraction → Chunking → Embeddings → ChromaDB

User Question → FastAPI → Vector Search + BM25 → RRF → Context Construction → Gemini → Answer + Sources → Streamlit

## 6. Functional Requirements

### FR-01: PDF Upload

- The system shall allow users to upload PDF files.
- The system shall validate the uploaded file type.
- The system shall reject unsupported file formats.
- The system shall handle empty or unreadable PDFs gracefully.

### FR-02: PDF Text Extraction

- The system shall extract selectable text from PDF pages.
- The system shall preserve original page numbers.
- The system shall identify pages where text extraction fails or returns empty content.

Note: OCR support for scanned PDFs is outside the initial release.

### FR-03: Text Chunking

- The system shall divide extracted text into smaller chunks.
- The system shall support configurable chunk size and overlap.
- Each chunk shall retain its source filename and original page number.
- Each chunk shall have a unique identifier.

### FR-04: Embedding Generation

- The system shall generate embeddings using Gemini.
- The system shall support multiple document chunks.
- The system shall handle API rate limits and temporary failures.
- The system shall avoid unnecessary repeated embedding generation.

### FR-05: Vector Storage

- The system shall store document chunks and embeddings in ChromaDB.
- The system shall associate metadata with each stored chunk.
- The system shall support retrieving relevant chunks using semantic similarity.

### FR-06: Hybrid Retrieval

- The system shall perform semantic retrieval using ChromaDB.
- The system shall perform keyword retrieval using BM25.
- The system shall combine retrieval rankings using RRF.
- The system shall return the most relevant document chunks.

### FR-07: Answer Generation

- The system shall construct context using retrieved chunks.
- The system shall send the context and user question to Gemini.
- The system shall generate answers based on the supplied context.
- The system shall handle insufficient or unavailable information.
- The system shall minimize unsupported or fabricated responses.

### FR-08: Source References

- The system shall return source document filenames.
- The system shall return original PDF page numbers.
- The system shall associate retrieved chunks with their source metadata.

### FR-09: User Interface

- The system shall provide an interactive document upload interface.
- The system shall provide a conversational question-answering interface.
- The system shall display generated responses.
- The system shall display source references alongside answers.
- The system shall display appropriate error messages.

## 7. Non-Functional Requirements

### Performance
- The system should process documents efficiently.
- Embedding generation should support batching where available.
- Retrieval should return results within a reasonable response time.

### Reliability
- The application should handle API failures gracefully.
- The application should validate incoming requests.
- The application should prevent duplicate document ingestion where practical.

### Security
- API keys must be stored securely using environment variables or secret management.
- API keys must never be exposed through frontend responses.
- Uploaded files should be validated before processing.
- User documents should not be exposed to other users.

### Maintainability
- Backend logic should be organized into separate modules.
- API routes, document processing, retrieval and generation should be separated.
- Code should include meaningful comments and documentation.

## 8. API Requirements

The FastAPI backend will expose the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload and process a PDF |
| GET | `/documents` | Retrieve uploaded document information |
| DELETE | `/documents/{document_id}` | Delete a document and its stored chunks |
| POST | `/chat/ask` | Ask a question about uploaded documents |
| GET | `/health` | Check backend availability |

### Example: Upload Document

**Endpoint:** `POST /documents/upload`

Input:
- PDF file using multipart/form-data.

Expected response:

{
  "message": "Document processed successfully",
  "document_id": "doc_001",
  "filename": "interview_guide.pdf",
  "total_pages": 35,
  "total_chunks": 110
}

### Example: Ask a Question

**Endpoint:** `POST /chat/ask`

Request:

{
  "document_id": "doc_001",
  "question": "Explain CPU scheduling algorithms"
}

Expected response:

{
  "answer": "CPU scheduling algorithms include FCFS, SJF, Round Robin and Priority Scheduling.",
  "sources": [
    {
      "filename": "interview_guide.pdf",
      "page": 4
    }
  ]
}

These are illustrative response structures; actual answers and source references will depend on the uploaded PDF.

## 9. Project Scope

### Version 1 – Minimum Viable Product (MVP)

The initial release will include:

- Single PDF upload.
- PDF text extraction.
- Text chunking with overlap.
- Gemini embedding generation.
- ChromaDB storage.
- BM25 keyword retrieval.
- Semantic vector retrieval.
- RRF-based hybrid search.
- Gemini answer generation.
- Source page references.
- Streamlit interface.
- FastAPI backend.

### Version 2 – Future Enhancements

The following features may be implemented later:

- Multiple PDF uploads.
- Conversational memory.
- Follow-up questions.
- OCR support for scanned PDFs.
- Persistent vector storage.
- Document management.
- Retrieval and answer evaluation.
- Improved citation verification.
- Authentication and user-specific document access.
- LangChain and LangGraph integration.

## 10. Project Folder Structure

Proposed structure:

DocuMind/

    backend/
        app/
            main.py

            routes/
                documents.py
                chat.py

            services/
                pdf_service.py
                chunking_service.py
                embedding_service.py
                vector_store.py
                retrieval_service.py
                rag_service.py

            schemas/
                document.py
                chat.py

            config.py

        requirements.txt
        .env.example
        .gitignore

    frontend/
        app.py
        requirements.txt

    README.md

## 11. Testing Requirements

The application shall be tested using the following scenarios:

1. Upload a valid text-based PDF.
2. Upload an unsupported file format.
3. Ask a question whose answer exists in the document.
4. Ask a question whose answer is unavailable.
5. Ask a question requiring information from multiple pages.
6. Verify that returned page references are correct.
7. Test API behavior when Gemini quota limits are reached.
8. Verify that document deletion removes associated stored chunks.

## 12. Acceptance Criteria

The MVP will be considered complete when:

- A user can upload a valid PDF.
- The backend extracts its text and page information.
- The system creates chunks and embeddings successfully.
- Chunks and metadata are stored in ChromaDB.
- BM25 and vector search work together.
- RRF combines retrieval results.
- Gemini generates answers using retrieved context.
- The application returns source page references.
- Unsupported questions receive an appropriate response.
- Users can interact with the system through Streamlit.
- The backend endpoints work correctly through Swagger UI.

## 13. Expected Outcome

DocuMind will demonstrate the practical implementation of a Retrieval-Augmented Generation system.

The project will showcase knowledge of:

- Large Language Model integration.
- Embedding generation.
- Vector databases.
- Semantic search.
- Keyword-based retrieval.
- Hybrid search.
- Reciprocal Rank Fusion.
- Context construction.
- Grounded answer generation.
- Backend API development.
- AI application integration.

## 14. Future Direction

After completing the MVP, DocuMind can be extended into a more advanced AI document assistant with conversational memory, multiple-document retrieval, evaluation mechanisms and agent-based workflows.

The long-term objective is to transform the application from a basic document question-answering system into a reliable and extensible AI-powered document assistant.
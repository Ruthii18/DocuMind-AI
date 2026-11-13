import uuid
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentInfo
)
from app.services.pdf_service import PDFService, PDFExtractionError
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Singleton service instances for dependency injection
_pdf_service = PDFService()
_chunking_service = ChunkingService()
_embedding_service = EmbeddingService()
_vector_store = VectorStoreService()


def get_pdf_service() -> PDFService:
    return _pdf_service


def get_chunking_service() -> ChunkingService:
    return _chunking_service


def get_embedding_service() -> EmbeddingService:
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    return _vector_store


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    pdf_service: PDFService = Depends(get_pdf_service),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
):
    """
    Upload a PDF document, extract its text, generate chunks and embeddings,
    and persist into ChromaDB.
    """
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF (.pdf) files are accepted."
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(e)}"
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )

    # 1. Extract text
    try:
        pages = pdf_service.extract_text_from_pdf(contents, filename)
    except PDFExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during PDF extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )

    document_id = f"doc_{uuid.uuid4().hex[:8]}"

    # 2. Chunk text
    chunks = chunking_service.chunk_document(
        document_id=document_id,
        filename=filename,
        pages=pages
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid text chunks could be created from this document."
        )

    # 3. Generate embeddings
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.get_embeddings_batch(chunk_texts)

    # 4. Store in ChromaDB
    try:
        vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error storing chunks in vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document in vector store: {str(e)}"
        )

    return DocumentUploadResponse(
        message="Document processed successfully",
        document_id=document_id,
        filename=filename,
        total_pages=len(pages),
        total_chunks=len(chunks),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Retrieve all indexed documents in the system."""
    docs = vector_store.list_documents()
    doc_infos = [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            total_pages=d["total_pages"],
            total_chunks=d["total_chunks"]
        )
        for d in docs
    ]
    return DocumentListResponse(documents=doc_infos, count=len(doc_infos))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Delete a document and its stored chunks from ChromaDB."""
    deleted_count = vector_store.delete_document(document_id)
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )

    return DocumentDeleteResponse(
        message=f"Document {document_id} and its {deleted_count} chunks deleted successfully.",
        document_id=document_id
    )


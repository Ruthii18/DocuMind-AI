import logging
from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.routes.documents import get_vector_store, get_embedding_service
from app.services.vector_store import VectorStoreService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_retrieval_service(
    vector_store: VectorStoreService = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> RetrievalService:
    return RetrievalService(
        vector_store=vector_store,
        embedding_service=embedding_service
    )


def get_rag_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> RAGService:
    return RAGService(retrieval_service=retrieval_service)


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Ask a question regarding uploaded documents.
    Retrieves context using hybrid search (Semantic + BM25 with RRF)
    and generates an answer grounded in the document content.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        response = rag_service.generate_answer(
            question=request.question,
            document_id=request.document_id
        )
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )


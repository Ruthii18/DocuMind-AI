import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.config import settings
from app.services.vector_store import VectorStoreService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """Basic lowercased word tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        rrf_k: int = settings.RRF_K,
        top_k: int = settings.TOP_K_RETRIEVAL
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.top_k = top_k

    def hybrid_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search:
        1. Semantic retrieval via ChromaDB.
        2. Keyword retrieval via BM25.
        3. Reciprocal Rank Fusion (RRF) to merge and re-rank.
        """
        k = top_k or self.top_k

        # 1. Semantic Retrieval
        vector_results = self._semantic_search(query, document_id=document_id, limit=k * 2)

        # 2. Keyword Retrieval (BM25)
        bm25_results = self._bm25_search(query, document_id=document_id, limit=k * 2)

        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=k
        )

        return fused_results

    def _semantic_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        query_embedding = self.embedding_service.get_embedding(query)
        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=limit,
            document_id=document_id
        )

    def _bm25_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword retrieval over stored chunks."""
        chunks = self.vector_store.get_all_chunks(document_id=document_id)
        if not chunks:
            return []

        tokenized_corpus = [simple_tokenize(c["content"]) for c in chunks]
        tokenized_query = simple_tokenize(query)

        if not tokenized_query:
            return chunks[:limit]

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.warning(f"BM25 ranking error: {e}. Falling back to default ordering.")
            return chunks[:limit]

        # Combine chunks with scores
        scored_chunks = []
        for chunk, score in zip(chunks, doc_scores):
            chunk_copy = dict(chunk)
            chunk_copy["bm25_score"] = float(score)
            scored_chunks.append(chunk_copy)

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_chunks[:limit]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum_{m in models} 1 / (rrf_k + rank_{m}(d))
        where rank starts at 1.
        """
        scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}

        # Process vector rankings
        for rank, chunk in enumerate(vector_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk

        # Process BM25 rankings
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk

        # Sort chunk IDs by aggregated RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        final_results = []
        for cid in sorted_ids[:top_k]:
            item = dict(chunk_data[cid])
            item["rrf_score"] = scores[cid]
            final_results.append(item)

        return final_results


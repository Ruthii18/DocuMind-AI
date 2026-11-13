import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.schemas.document import DocumentChunk, DocumentInfo

logger = logging.getLogger(__name__)


class VectorStoreService:
    COLLECTION_NAME = "documind_chunks"

    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB at {self.persist_dir} with collection '{self.COLLECTION_NAME}'.")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add document chunks and their precomputed embeddings to ChromaDB."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query top-k most similar chunks using cosine similarity.
        Optionally filter by document_id.
        """
        where_filter = {"document_id": document_id} if document_id else None

        # Chroma query requires n_results <= total count in collection
        total_count = self.collection.count()
        if total_count == 0:
            return []

        actual_k = min(top_k, total_count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else 0.0
            # Cosine distance to similarity: similarity = 1 - distance
            score = 1.0 - dist
            formatted_results.append({
                "chunk_id": ids[i],
                "content": docs[i] if i < len(docs) else "",
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score
            })

        return formatted_results

    def get_all_chunks(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks, optionally filtered by document_id."""
        where_filter = {"document_id": document_id} if document_id else None
        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        formatted_results: List[Dict[str, Any]] = []
        if not results or not results["ids"]:
            return formatted_results

        ids = results["ids"]
        docs = results["documents"] or []
        metas = results["metadatas"] or []

        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            formatted_results.append({
                "chunk_id": ids[i],
                "content": docs[i] if i < len(docs) else "",
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
            })

        return formatted_results

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document_id."""
        chunks = self.get_all_chunks(document_id=document_id)
        if not chunks:
            return 0

        chunk_ids = [c["chunk_id"] for c in chunks]
        self.collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks for document_id={document_id}.")
        return len(chunk_ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents stored in ChromaDB."""
        all_chunks = self.get_all_chunks()
        doc_map: Dict[str, Dict[str, Any]] = {}

        for chunk in all_chunks:
            doc_id = chunk["document_id"]
            if not doc_id:
                continue

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": chunk["filename"],
                    "total_chunks": 0,
                    "pages": set(),
                }
            doc_map[doc_id]["total_chunks"] += 1
            doc_map[doc_id]["pages"].add(chunk["page_number"])

        documents = []
        for doc_id, info in doc_map.items():
            documents.append({
                "document_id": doc_id,
                "filename": info["filename"],
                "total_chunks": info["total_chunks"],
                "total_pages": len(info["pages"]),
            })

        return documents


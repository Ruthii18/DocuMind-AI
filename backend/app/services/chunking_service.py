from typing import List, Dict, Any
from app.config import settings
from app.schemas.document import DocumentChunk


class ChunkingService:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Divide extracted page texts into overlapping chunks, maintaining
        metadata for document_id, filename, and original page number.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_num = page["page_number"]
            text = page.get("text", "").strip()
            if not text:
                continue

            page_chunks = self._split_text(text)
            for chunk_content in page_chunks:
                chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_number=page_num,
                        content=chunk_content,
                        chunk_index=global_chunk_idx,
                    )
                )
                global_chunk_idx += 1

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks respecting word boundaries where possible.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find closest word boundary before the end limit
            boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start index forward, considering overlap
            step = max(1, (end - start) - self.chunk_overlap)
            start += step

        return chunks


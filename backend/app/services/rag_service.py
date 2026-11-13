import logging
from typing import List, Dict, Any
from app.config import settings
from app.schemas.chat import ChatResponse, SourceReference
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are DocuMind, an intelligent and precise AI assistant specializing in document question answering using Retrieval-Augmented Generation (RAG).

Instructions:
1. Answer the user's question based ONLY on the provided context retrieved from the uploaded document(s).
2. If the context does not contain enough information to answer the question accurately, clearly state:
   "I'm sorry, but the uploaded document does not contain sufficient information to answer this question."
3. Do NOT make up facts or extrapolate beyond the provided text chunks.
4. Keep the answer direct, informative, structured, and easy to read.
5. Whenever relevant, refer to specific details found in the context.
"""


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        api_key: str = settings.GEMINI_API_KEY,
        model_name: str = settings.LLM_MODEL
    ):
        self.retrieval_service = retrieval_service
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        self._sdk_type = None

        if self.api_key:
            self._init_llm_client()
        else:
            logger.warning("GEMINI_API_KEY is not configured. RAG answer generation will run in fallback mode.")

    def _init_llm_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
            logger.info("Initialized Google GenAI LLM client.")
            return
        except ImportError:
            pass

        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy
            self._sdk_type = "legacy"
            logger.info("Initialized legacy Google Generative AI LLM client.")
            return
        except ImportError:
            pass

        logger.warning("No Google GenAI package installed. LLM client unavailable.")

    def generate_answer(
        self,
        question: str,
        document_id: str = None
    ) -> ChatResponse:
        """
        Execute RAG workflow:
        1. Hybrid retrieve top-k chunks.
        2. Construct prompt context.
        3. Call Gemini LLM to generate grounded answer.
        4. Extract unique source references.
        """
        retrieved_chunks = self.retrieval_service.hybrid_search(
            query=question,
            document_id=document_id
        )

        if not retrieved_chunks:
            return ChatResponse(
                answer="No relevant documents or text chunks were found. Please upload a document first.",
                sources=[]
            )

        # Build context
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            block = (
                f"[Source Chunk {i}] (Document: {chunk['filename']}, Page: {chunk['page_number']})\n"
                f"{chunk['content']}"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        user_prompt = f"""DOCUMENT CONTEXT:
---------------------
{context_text}
---------------------

USER QUESTION:
{question}

Please provide a helpful and accurate answer grounded strictly in the DOCUMENT CONTEXT above:"""

        # Generate answer with LLM
        answer = self._call_llm(user_prompt)

        # Build deduplicated sources
        sources: List[SourceReference] = []
        seen_pages = set()

        for chunk in retrieved_chunks:
            key = (chunk["filename"], chunk["page_number"])
            if key not in seen_pages:
                seen_pages.add(key)
                # Take first 150 chars as snippet preview
                snippet = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                sources.append(
                    SourceReference(
                        filename=chunk["filename"],
                        page=chunk["page_number"],
                        snippet=snippet
                    )
                )

        return ChatResponse(
            answer=answer,
            sources=sources
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Gemini LLM with error handling and fallback."""
        if not self._client or not self.api_key:
            return (
                "⚠️ Notice: GEMINI_API_KEY is not configured or client could not be loaded. "
                "Retrieved relevant chunks successfully. Please set your GEMINI_API_KEY in backend/.env "
                "to enable automated Gemini answer generation."
            )

        try:
            if self._sdk_type == "genai":
                clean_model = self.model_name.replace("models/", "")
                response = self._client.models.generate_content(
                    model=clean_model,
                    contents=[SYSTEM_PROMPT, prompt]
                )
                return response.text.strip() if response.text else "No response generated by model."

            elif self._sdk_type == "legacy":
                model = self._client.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                response = model.generate_content(prompt)
                return response.text.strip() if response.text else "No response generated by model."

        except Exception as e:
            logger.error(f"Error calling Gemini LLM: {e}")
            return f"Error generating answer from Gemini: {str(e)}"


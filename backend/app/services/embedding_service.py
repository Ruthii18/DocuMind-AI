import logging
import hashlib
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        self._sdk_type = None

        if self.api_key:
            self._init_client()
        else:
            logger.warning("GEMINI_API_KEY is not configured. Embeddings will use local fallback representations.")

    def _init_client(self):
        try:
            # Try new google-genai client first
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
            logger.info("Initialized Google GenAI client for embeddings.")
            return
        except ImportError:
            pass

        try:
            # Try google-generativeai client
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy
            self._sdk_type = "legacy"
            logger.info("Initialized legacy Google Generative AI client for embeddings.")
            return
        except ImportError:
            pass

        logger.warning("No Google GenAI package installed. Running in mock embedding mode.")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0] if embeddings else self._mock_embedding(text)

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings for a list of strings with batching and fallback."""
        if not texts:
            return []

        # If no client or no key, return mock embedding (deterministic vector from hash)
        if not self._client or not self.api_key:
            return [self._mock_embedding(t) for t in texts]

        results: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                if self._sdk_type == "genai":
                    # Using new google.genai
                    # Strip 'models/' prefix if present for new SDK
                    clean_model = self.model_name.replace("models/", "")
                    response = self._client.models.embed_content(
                        model=clean_model,
                        contents=batch
                    )
                    for emb in response.embeddings:
                        results.append(emb.values)

                elif self._sdk_type == "legacy":
                    # Using legacy google.generativeai
                    model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
                    response = self._client.embed_content(
                        model=model_path,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    # response['embedding'] is a list of embeddings for batch
                    if "embedding" in response:
                        if isinstance(response["embedding"][0], list):
                            results.extend(response["embedding"])
                        else:
                            results.append(response["embedding"])
                    else:
                        raise ValueError("Unexpected embedding response structure")
            except Exception as e:
                logger.error(f"Error calling Gemini Embedding API for batch {i}: {e}. Falling back to mock embeddings.")
                for text in batch:
                    results.append(self._mock_embedding(text))

        return results

    def _mock_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """
        Generate deterministic normalized pseudo-embedding based on text hash
        for testing or when API key is unavailable.
        """
        seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(dimension):
            byte_val = seed_hash[i % len(seed_hash)]
            vector.append((byte_val / 128.0) - 1.0)
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]


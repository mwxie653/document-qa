"""Cross-Encoder reranker for improving retrieval precision.

After Chroma retrieves top-k*2 chunks, the reranker scores each (query, chunk)
pair and returns the top-k most relevant ones. This is a standard RAG optimization:
coarse retrieval (bi-encoder) -> fine re-ranking (cross-encoder).

Model: BAAI/bge-reranker-base — supports Chinese and English, widely used.
"""

import os

# 国内用户：自动切到 HuggingFace 镜像
if os.getenv("HF_MIRROR") == "1" and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import CrossEncoder  # noqa: E402

MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    """Cross-Encoder based reranker. Lazy-loads the model."""

    def __init__(self):
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            try:
                self._model = CrossEncoder(MODEL_NAME)
            except OSError:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._model = CrossEncoder(MODEL_NAME, local_files_only=True)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Re-rank chunks by cross-encoder relevance scores.

        Args:
            query: User question
            chunks: Candidate chunk texts from coarse retrieval
            top_k: Number of top chunks to return after re-ranking

        Returns:
            [{text, score}, ...] sorted by relevance descending
        """
        if not chunks:
            return []

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by score descending
        ranked = sorted(
            zip(chunks, scores.tolist()), key=lambda x: x[1], reverse=True
        )

        return [
            {"text": chunk, "score": round(score, 4)}
            for chunk, score in ranked[:top_k]
        ]

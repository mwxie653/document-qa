"""Embedding model wrapper using sentence-transformers.

Uses paraphrase-multilingual-MiniLM-L12-v2 (384-dim, Chinese + English).

国内用户：在 .env 中设置 HF_MIRROR=1，使用 hf-mirror.com 下载模型。
首次运行会下载模型（约 470MB），后续自动使用缓存。
"""

import os

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 国内用户：自动切到 HuggingFace 镜像
if os.getenv("HF_MIRROR") == "1" and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer  # noqa: E402


class EmbeddingModel:
    """Lazy-loads and caches the sentence-transformers model."""

    def __init__(self):
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(MODEL_NAME)
            except OSError:
                # 网络不通时尝试用本地缓存
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts. Returns [[float, ...], ...]."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]

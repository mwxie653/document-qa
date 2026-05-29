"""RAG engine orchestrating document parsing, embedding, storage, and query."""

from src.document_parser import parse_document
from src.chunker import chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.deepseek_client import DeepSeekClient
from src.reranker import Reranker

SYSTEM_PROMPT = """你是一个专业的文档问答助手。根据用户提供的文档片段回答问题。

要求：
1. 只根据提供的文档内容作答，不要编造信息。
2. 如果文档中没有相关信息，直接说"文档中未找到相关信息"。
3. 回答中引用原文时，使用【引用】标记来源段落。
4. 用中文回答，保持简洁清晰。"""


class RAGEngine:
    """Orchestrates the full RAG pipeline: index documents and answer questions."""

    def __init__(
        self,
        deepseek_api_key: str | None = None,
        persist_dir: str = "./data/chroma",
        use_reranker: bool = False,
    ):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.llm = DeepSeekClient(api_key=deepseek_api_key)
        self.use_reranker = use_reranker
        self.reranker = Reranker() if use_reranker else None

    def index_document(self, file_bytes: bytes, filename: str) -> str:
        """Process and store a document. Returns the collection ID."""
        text = parse_document(file_bytes, filename)
        chunks = chunk_text(text)
        embeddings = self.embedder.embed(chunks)
        collection_id = self.vector_store.add_documents(chunks, embeddings, filename)
        return collection_id

    def query(self, question: str, collection_id: str, top_k: int = 5) -> dict:
        """Answer a question using the indexed document. Returns {answer, sources}."""
        q_embedding = self.embedder.embed_query(question)

        # 粗检索：从向量库多拿一些候选
        fetch_k = top_k * 3 if self.use_reranker else top_k
        sources = self.vector_store.search(q_embedding, collection_id, top_k=fetch_k)

        if not sources:
            return {"answer": "未找到相关文档内容，请先上传并索引文档。", "sources": []}

        # 重排序：用 Cross-Encoder 精排
        if self.use_reranker and len(sources) > top_k:
            chunk_texts = [s["text"] for s in sources]
            reranked = self.reranker.rerank(question, chunk_texts, top_k=top_k)
            # Rebuild sources from reranked results, preserving metadata by index
            reranked_sources = []
            for r in reranked:
                for s in sources:
                    if s["text"] == r["text"]:
                        s_copy = dict(s)
                        s_copy["rerank_score"] = r["score"]
                        reranked_sources.append(s_copy)
                        break
            sources = reranked_sources

        context_parts = []
        for i, s in enumerate(sources):
            context_parts.append(f"[片段{i+1}]\n{s['text']}")

        context = "\n\n---\n\n".join(context_parts)
        user_prompt = f"文档内容：\n{context}\n\n用户问题：{question}"

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return {"answer": answer, "sources": sources}

    def list_documents(self) -> list[dict]:
        """List all indexed documents."""
        return self.vector_store.list_collections()

    def delete_document(self, collection_id: str):
        """Remove an indexed document."""
        self.vector_store.delete_collection(collection_id)

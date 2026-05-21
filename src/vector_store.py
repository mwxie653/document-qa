"""ChromaDB vector store wrapper for document chunk storage and retrieval."""

import uuid
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Manages ChromaDB collections for document chunks."""

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        doc_name: str,
    ) -> str:
        """Store chunks with embeddings. Returns the collection name."""
        collection_id = f"doc-{uuid.uuid4().hex[:8]}"
        collection = self.client.create_collection(name=collection_id)

        ids = [f"{collection_id}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_name": doc_name, "chunk_index": i, "text_preview": chunks[i][:200]}
            for i in range(len(chunks))
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return collection_id

    def search(
        self,
        query_embedding: list[float],
        collection_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for the most similar chunks. Returns [{text, metadata, distance}, ...]."""
        collection = self.client.get_collection(name=collection_id)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        sources = []
        if results["ids"][0]:
            for i in range(len(results["ids"][0])):
                sources.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        return sources

    def delete_collection(self, collection_id: str):
        """Remove a document collection by name."""
        try:
            self.client.delete_collection(name=collection_id)
        except Exception:
            pass

    def list_collections(self) -> list[dict]:
        """List all indexed document collections."""
        cols = []
        for c in self.client.list_collections():
            cols.append({"name": c.name, "count": c.count()})
        return cols

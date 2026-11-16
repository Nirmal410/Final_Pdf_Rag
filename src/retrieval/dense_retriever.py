
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_store import ChromaStore
from config.settings import TOP_K_DENSE


class DenseRetriever:

    def __init__(self, embedding_service=None, store=None):

        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

        self.store = store or ChromaStore()

    def retrieve(self, query, top_k=TOP_K_DENSE):

        query_embedding = (
            self.embedding_service.embed_query(query)
        )

        results = self.store.search(
            query_embedding,
            top_k=top_k
        )

        retrieved = []

        if not results["ids"]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):

            retrieved.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved

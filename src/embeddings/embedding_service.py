
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


class EmbeddingService:

    QUERY_PREFIX = (
        "Represent this sentence for searching "
        "relevant passages: "
    )

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    def embed_documents(self, texts: list[str]):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_query(self, query: str):

        query = self.QUERY_PREFIX + query

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding.tolist()

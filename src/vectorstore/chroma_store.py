
import chromadb

from config.settings import CHROMA_DIR, COLLECTION_NAME


class ChromaStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings count mismatch"
            )

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[
                chunk["text"] for chunk in chunks
            ],
            embeddings=embeddings,
            metadatas=[
                chunk["metadata"] for chunk in chunks
            ]
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def clear(self):

        self.client.delete_collection(
            COLLECTION_NAME
        )

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_store import ChromaStore

embedding_service = EmbeddingService()

texts = [
    "Java is an object-oriented programming language.",
    "Python is commonly used in machine learning."
]

vectors = embedding_service.embed_documents(texts)

store = ChromaStore()

chunks = [
    {
        "text": text,
        "metadata": {
            "source": "test.pdf",
            "page": i + 1,
            "chunk_index": 0
        }
    }
    for i, text in enumerate(texts)
]

store.add_documents(chunks, vectors)

query_vector = embedding_service.embed_query(
    "What is Java?"
)

results = store.search(query_vector, top_k=2)

print(results)
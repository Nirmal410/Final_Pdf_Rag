
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.compressor import ContextCompressor

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_store import ChromaStore

from config.settings import FINAL_TOP_K


class RetrievalService:

    def __init__(self, chunks):

        self.embedding_service = EmbeddingService()
        self.store = ChromaStore()

        embeddings = self.embedding_service.embed_documents(
            [chunk["text"] for chunk in chunks]
        )

        self.store.add_documents(chunks, embeddings)

        self.dense = DenseRetriever(
            self.embedding_service,
            self.store
        )

        self.bm25 = BM25Retriever()
        self.bm25.build_index(chunks)

        self.hybrid = HybridRetriever(
            self.dense,
            self.bm25
        )

        self.multi_query = MultiQueryRetriever(
            self.hybrid
        )

        self.reranker = Reranker()

        self.compressor = ContextCompressor(
            self.embedding_service
        )

    def retrieve(self, question, history=None):

        candidates = self.multi_query.retrieve(
            question,
            top_k=10,
            history=history
        )

        reranked = self.reranker.rerank(
            question,
            candidates,
            top_k=5
        )

        compressed = self.compressor.compress(
            question,
            reranked    
        )

        return compressed[:FINAL_TOP_K]

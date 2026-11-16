
from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL,
            device="cpu"
        )

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        ranked_documents = []

        for document, score in zip(
            documents, scores
        ):

            item = document.copy()

            item["rerank_score"] = float(score)

            ranked_documents.append(item)

        ranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked_documents[:top_k]

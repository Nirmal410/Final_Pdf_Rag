
import re
from rank_bm25 import BM25Okapi
from config.settings import TOP_K_BM25


def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


class BM25Retriever:

    def __init__(self):

        self.documents = []
        self.bm25 = None

    def build_index(self, chunks):

        self.documents = chunks

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        if not tokenized_documents:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def retrieve(self, query, top_k=TOP_K_BM25):

        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.documents[index]

            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "bm25_score": score
            })

        return results

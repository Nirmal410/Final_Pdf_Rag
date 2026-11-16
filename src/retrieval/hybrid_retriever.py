
from collections import defaultdict


class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        rrf_k=60
    ):

        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(self, result_lists):

        fused_scores = defaultdict(float)
        documents = {}

        for result_list in result_lists:

            for rank, document in enumerate(
                result_list, start=1
            ):

                doc_id = document["id"]

                fused_scores[doc_id] += (
                    1 / (self.rrf_k + rank)
                )

                documents[doc_id] = document

        ranked_ids = sorted(
            fused_scores,
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True
        )

        results = []

        for doc_id in ranked_ids:

            document = documents[doc_id].copy()

            document["rrf_score"] = (
                fused_scores[doc_id]
            )

            results.append(document)

        return results

    def retrieve(self, query, top_k=10):

        try:
            dense_results = self.dense.retrieve(query, top_k=top_k)
        except TypeError:
            dense_results = self.dense.retrieve(query)

        try:
            bm25_results = self.bm25.retrieve(query, top_k=top_k)
        except TypeError:
            bm25_results = self.bm25.retrieve(query)

        results = self.reciprocal_rank_fusion([
            dense_results,
            bm25_results
        ])

        return results[:top_k]

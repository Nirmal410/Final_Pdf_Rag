
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeRetriever:

    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=None):
        return self.results[:top_k] if top_k else self.results


def test_rrf_combines_results():

    dense_results = [
        {"id": "A", "text": "Java"},
        {"id": "B", "text": "Python"}
    ]

    bm25_results = [
        {"id": "B", "text": "Python"},
        {"id": "C", "text": "Spring"}
    ]

    hybrid = HybridRetriever(
        FakeRetriever(dense_results),
        FakeRetriever(bm25_results)
    )

    results = hybrid.retrieve("Java", top_k=3)

    assert len(results) == 3

    # B appears near the top in both result lists.
    assert results[0]["id"] == "B"

    assert "rrf_score" in results[0]


from src.pipeline.rag_pipeline import RAGPipeline


class FakeRetrieval:

    def retrieve(self, question, history=None):

        return [
            {
                "id": "test_1",
                "text": "Java supports automatic garbage collection.",
                "metadata": {
                    "source": "java.pdf",
                    "page": 5
                }
            }
        ]


class FakeLLM:

    def generate(self, prompt):

        assert "garbage collection" in prompt

        return (
            "Java supports automatic garbage "
            "collection. [Source 1]"
        )


def test_rag_answer():

    pipeline = RAGPipeline(
        FakeRetrieval()
    )

    # Replace actual model with a fake for this test.
    pipeline.llm = FakeLLM()

    result = pipeline.answer(
        "What is garbage collection?"
    )

    assert "automatic garbage collection" in result["answer"]

    assert result["sources"][0]["source"] == "java.pdf"

    assert result["sources"][0]["page"] == 5


import json
import re
from ollama import Client
from config.settings import (
    LLM_MODEL,
    OLLAMA_BASE_URL
)


class MultiQueryRetriever:

    def __init__(self, hybrid_retriever):

        self.hybrid = hybrid_retriever

        self.client = Client(
            host=OLLAMA_BASE_URL
        )

    def generate_queries(self, question):

        prompt = f"""
Generate 3 different search queries for the
following question.

Keep the original meaning.

Return only a JSON array of strings.
Do not answer the question.

Question: {question}
"""

        try:
            response = self.client.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json",
                options={"temperature": 0}
            )

            content = response["message"]["content"]
            data = json.loads(content)

            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict):
                queries = data.get("queries", [])
            else:
                queries = []

        except Exception:
            queries = []

        if not isinstance(queries, list):
            queries = []

        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str) and q.strip()
        ]

        # Always include the original question.
        return list(dict.fromkeys(
            [question] + queries[:3]
        ))

    def retrieve(self, question, top_k=10, history=None):

        question = self.rewrite_question(
        question,
        history
        )

        queries = self.generate_queries(question)

        result_lists = []

        for query in queries:

            results = self.hybrid.retrieve(
                query,
                top_k=top_k
            )

            result_lists.append(results)

        combined = self.hybrid.reciprocal_rank_fusion(
            result_lists
        )

        return combined[:top_k]
    
    def rewrite_question(self, question, history=None):

        if not history:
            return question

        recent_history = history[-4:]

        conversation = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
        )

        prompt = f"""
Rewrite the latest question as a standalone
search query using the conversation history.

Do not answer the question.
Return only the rewritten query.

Conversation:
{conversation}

Latest question:
{question}
"""

        try:
            response = self.client.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={"temperature": 0}
            )

            rewritten = response["message"]["content"].strip()
            return rewritten or question
        except Exception:
            return question


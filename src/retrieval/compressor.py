
import re
import numpy as np


class ContextCompressor:

    def __init__(self, embedding_service):

        self.embedding_service = embedding_service

    def compress(
        self,
        query,
        documents,
        similarity_threshold=0.30
    ):

        compressed = []

        query_embedding = np.array(
            self.embedding_service.embed_query(query)
        )

        for document in documents:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                document["text"]
            )

            sentences = [
                sentence.strip()
                for sentence in sentences
                if len(sentence.split()) >= 4
            ]

            if not sentences:
                continue

            sentence_embeddings = np.array(
                self.embedding_service.embed_documents(
                    sentences
                )
            )

            similarities = (
                sentence_embeddings @ query_embedding
            )

            relevant_sentences = [
                sentence
                for sentence, score in zip(
                    sentences, similarities
                )
                if score >= similarity_threshold
            ]

            if relevant_sentences:

                item = document.copy()

                item["original_text"] = document["text"]

                item["text"] = " ".join(
                    relevant_sentences
                )

                compressed.append(item)

        # Fallback to original documents if compression filtered out all sentences
        return compressed if compressed else documents

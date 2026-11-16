
from src.generation.llm_service import LLMService
from src.generation.prompt_templates import build_prompt
from config.settings import FINAL_TOP_K


class RAGPipeline:

    def __init__(self, retrieval_service):

        self.retrieval = retrieval_service
        self.llm = LLMService()

    
    def format_context(self, documents):

        context_parts = []

        for index, document in enumerate(documents, start=1):

            metadata = document["metadata"]

            context_parts.append(
                f"""
[Source {index}]
Document: {metadata['source']}
Page: {metadata['page']}

Content:
{document['text']}
"""
            )

        return "\n\n".join(context_parts)


    def format_history(self, history):

        if not history:
            return ""

        messages = []

        for message in history[-4:]:

            role = message["role"]
            content = message["content"]

            messages.append(
                f"{role.upper()}: {content}"
            )

        return "\n".join(messages)

    def answer(self, question, history=None):

        history = history or []

        # Step 1: Retrieve relevant chunks
        documents = self.retrieval.retrieve(question,history=history)

        if not documents:

            return {
                "answer": (
                    "I could not find sufficient "
                    "information in the PDF."
                ),
                "sources": [],
                "retrieved_chunks": []
            }

        # Step 2: Prepare context
        context = self.format_context(documents)

        # Step 3: Prepare conversation history
        chat_history = self.format_history(history)

        # Step 4: Build grounded prompt
        prompt = build_prompt(
            question=question,
            context=context,
            chat_history=chat_history
        )

        # Step 5: Generate answer
        answer = self.llm.generate(prompt)

        # Step 6: Extract source references
        sources = []

        seen = set()

        for document in documents:

            metadata = document["metadata"]

            source_key = (
                metadata["source"],
                metadata["page"]
            )

            if source_key not in seen:

                seen.add(source_key)

                sources.append({
                    "source": metadata["source"],
                    "page": metadata["page"]
                })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": documents
        }

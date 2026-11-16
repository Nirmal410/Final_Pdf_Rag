
import hashlib

from src.ingestion.chunker import chunk_text
from src.ingestion.text_cleaner import clean_text


def create_chunks(documents):

    all_chunks = []

    for document in documents:

        text = clean_text(document["text"])
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            source = document["metadata"]["source"]
            page = document["metadata"]["page"]

            unique_value = (
                f"{source}_{page}_{index}_{chunk}"
            )

            chunk_id = hashlib.sha256(
                unique_value.encode("utf-8")
            ).hexdigest()

            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": index
                }
            })

    return all_chunks

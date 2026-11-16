
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str) -> list[str]:

    words = text.split()

    if not words:
        return []

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size"
        )

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks

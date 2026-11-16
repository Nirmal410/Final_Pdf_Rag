
from src.ingestion.chunker import chunk_text


def test_chunking():

    text = " ".join(
        [f"word{i}" for i in range(1000)]
    )

    chunks = chunk_text(text)

    assert len(chunks) > 1

    assert len(chunks[0].split()) == 350

    assert len(chunks[1].split()) == 350

    assert chunks[0].split()[-50:] == (
        chunks[1].split()[:50]
    )

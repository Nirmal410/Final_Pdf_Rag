
import fitz
from pathlib import Path


def load_pdf(pdf_path: str) -> list[dict]:

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    documents = []

    with fitz.open(pdf_path) as pdf:

        if pdf.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            if not text.strip():
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number
                }
            })

    if not documents:
        raise ValueError(
            "No extractable text found in PDF"
        )

    return documents

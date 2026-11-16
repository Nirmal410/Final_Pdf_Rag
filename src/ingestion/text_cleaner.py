
import re


def clean_text(text: str) -> str:

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()

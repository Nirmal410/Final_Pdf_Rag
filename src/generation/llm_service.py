
from ollama import Client

from config.settings import (
    LLM_MODEL,
    OLLAMA_BASE_URL
)


class LLMService:

    def __init__(self):

        self.client = Client(
            host=OLLAMA_BASE_URL
        )

    def generate(
        self,
        prompt,
        temperature=0.1
    ):

        try:

            response = self.client.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": temperature,
                    "num_ctx": 4096
                }
            )

            return response["message"]["content"]

        except Exception as error:

            raise RuntimeError(
                f"LLM generation failed: {error}"
            ) from error

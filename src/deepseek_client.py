"""DeepSeek API client via OpenAI-compatible interface."""

import os
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient:
    """Wraps the DeepSeek API for RAG answer generation."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is required. Set it in .env or pass directly.")
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model: str = "deepseek-chat",
    ) -> str:
        """Send prompt to DeepSeek and return the generated answer."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

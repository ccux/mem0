import os
from typing import Literal, Optional

from google import genai
from google.genai import types

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class GoogleGenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        # Use environment variable for model name, fallback to default
        self.config.model = self.config.model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.config.embedding_dims = self.config.embedding_dims or int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

        api_key = self.config.api_key or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")

        # Configure the client with API key
        self.client = genai.Client(api_key=api_key)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Google Generative AI.
        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")

        # Call the embed_content method with output_dimensionality parameter to ensure 1536 dimensions
        result = self.client.models.embed_content(
            model=self.config.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dims)
        )

        # Extract the embedding values from the result
        [embedding_obj] = result.embeddings
        return embedding_obj.values

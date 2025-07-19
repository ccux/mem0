from google import genai
from google.genai import types
from typing import List, Dict, Any
import logging
import numpy as np

from config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_LLM_MODEL, CATEGORIZATION_PROMPT, MEMORY_CATEGORIES, VECTOR_DIMENSION

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.embedding_model = GEMINI_EMBEDDING_MODEL
        self.llm_model = GEMINI_LLM_MODEL
        logger.info(f"Initialized Gemini client with embedding model: {self.embedding_model}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini."""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIMENSION)
            )
            [embedding_obj] = result.embeddings
            embedding_values_np = np.array(embedding_obj.values)
            # Only normalize if not 3072-dim (Gemini normalizes 3072-dim by default)
            if VECTOR_DIMENSION in (768, 1536):
                norm = np.linalg.norm(embedding_values_np)
                if norm > 0:
                    embedding_values_np = embedding_values_np / norm
            return embedding_values_np.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def categorize_memory(self, content: str) -> str:
        """Categorize memory content using Gemini LLM."""
        try:
            prompt = CATEGORIZATION_PROMPT.format(content=content)

            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            category = response.text.strip().lower()

            # Validate category
            if category in MEMORY_CATEGORIES:
                logger.info(f"Categorized content as: {category}")
                return category
            else:
                logger.warning(f"Invalid category '{category}', defaulting to 'general'")
                return "general"

        except Exception as e:
            logger.error(f"Error categorizing memory: {e}")
            return "general"  # Fallback to general category

    def extract_memories_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract memorable information from conversation messages."""
        try:
            # Combine all message content
            combined_content = []
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    combined_content.append(str(message["content"]))
                elif isinstance(message, str):
                    combined_content.append(message)

            if not combined_content:
                return []

            full_content = " ".join(combined_content)

            # Use LLM to extract memorable information
            extraction_prompt = f"""
            Extract key memorable information from the following conversation.
            Focus on facts, preferences, goals, important events, and personal details.
            Return each piece of information as a separate line.
            Only return factual, memorable information - ignore casual conversation.

            Conversation:
            {full_content}

            Memorable information (one per line):
            """

            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=extraction_prompt
            )

            # Parse response into individual memories
            memories = []
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('-') and len(line) > 10:
                    # Clean up the line
                    if line.startswith('- '):
                        line = line[2:]
                    memories.append(line)

            logger.info(f"Extracted {len(memories)} memories from messages")
            return memories

        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []

    def generate_response(self, messages, tools=None, tool_choice="auto"):
        """Generate response using Gemini LLM for compatibility with Mem0 interface."""
        try:
            # Convert messages to a simple prompt if needed
            if isinstance(messages, list):
                prompt = "\n".join([msg.get("content", str(msg)) for msg in messages])
            else:
                prompt = str(messages)

            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            return response.text if response else ""

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

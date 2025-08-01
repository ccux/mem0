import logging
from typing import Dict, Any, List
from config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_LLM_MODEL, KNOWLEDGE_EXTRACTION_PROMPT, MEMORY_CATEGORIES, VECTOR_DIMENSION
import google.generativeai as genai
try:
    from google import genai as google_genai
    from google.genai import types
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    logger = logging.getLogger(__name__)
    logger.warning("google-genai package not available, using google-generativeai without output_dimensionality")

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        """Initialize Gemini client with API key and models."""
        genai.configure(api_key=GOOGLE_API_KEY)
        # For embeddings, we use the embedding model directly, not GenerativeModel
        self.embedding_model_name = GEMINI_EMBEDDING_MODEL
        self.llm_model = genai.GenerativeModel(GEMINI_LLM_MODEL)
        self.client = genai
        
        # Initialize new API client if available
        if USE_NEW_API:
            self.google_client = google_genai.Client(api_key=GOOGLE_API_KEY)

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Gemini."""
        try:
            if USE_NEW_API:
                # Use the new Google GenAI client API with output_dimensionality
                result = self.google_client.models.embed_content(
                    model=self.embedding_model_name,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=VECTOR_DIMENSION)
                )
                # Extract embedding values from the result
                if result.embeddings:
                    embedding_obj = result.embeddings[0]
                    return list(embedding_obj.values)
                else:
                    logger.error("No embeddings returned from Gemini")
                    return [0.0] * VECTOR_DIMENSION
            else:
                # Fallback to google.generativeai without output_dimensionality
                # This will return 768 dimensions by default
                logger.warning("Using google-generativeai without output_dimensionality - dimension mismatch may occur")
                response = genai.embed_content(
                    model=self.embedding_model_name,
                    content=text
                )
                # Extract embedding from response
                if response and 'embedding' in response:
                    return response['embedding']
                else:
                    logger.error("No embedding returned from Gemini")
                    return [0.0] * VECTOR_DIMENSION
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return [0.0] * VECTOR_DIMENSION

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (alias for get_embeddings)."""
        return self.get_embeddings(text)

    def extract_memories_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract memories from a list of messages."""
        try:
            # Combine all messages into a single text
            combined_text = ""
            for message in messages:
                if isinstance(message, dict):
                    content = message.get('content', '')
                    if isinstance(content, str):
                        combined_text += content + "\n"
                elif isinstance(message, str):
                    combined_text += message + "\n"

            if not combined_text.strip():
                logger.warning("No content found in messages")
                return []

            # Use the sophisticated knowledge extraction prompt
            prompt = f"{KNOWLEDGE_EXTRACTION_PROMPT}\n\nContent to analyze:\n{combined_text}"

            response = self.llm_model.generate_content(prompt)
            response_text = response.text.strip()

            # Parse the JSON response
            import json
            import re
            try:
                # Clean the response text - remove markdown code blocks if present
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()

                result = json.loads(cleaned_response)

                if result.get("items") and len(result["items"]) > 0:
                    # Extract the memory text from each item
                    memories = []
                    for item in result["items"]:
                        # Try both "memory" and "content" fields
                        memory_text = item.get("memory", "") or item.get("content", "")
                        if memory_text and memory_text.strip():
                            memories.append(memory_text.strip())

                    logger.info(f"Extracted {len(memories)} memories from messages")
                    return memories
                else:
                    logger.info("No memories extracted from messages")
                    return []

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from memory extraction: {e}")
                logger.error(f"Raw response: {response_text}")
                return []

        except Exception as e:
            logger.error(f"Error extracting memories from messages: {e}")
            return []

    def categorize_memory_sophisticated(self, content: str) -> str:
        """Categorize memory content using sophisticated knowledge extraction."""
        try:
            # Use the sophisticated knowledge extraction prompt
            prompt = f"{KNOWLEDGE_EXTRACTION_PROMPT}\n\nContent to analyze:\n{content}"

            response = self.llm_model.generate_content(prompt)
            response_text = response.text.strip()

            # Parse the JSON response
            import json
            try:
                result = json.loads(response_text)

                if result.get("items") and len(result["items"]) > 0:
                    # Get the highest confidence item
                    best_item = max(result["items"], key=lambda x: x.get("confidence", 0))
                    category = best_item.get("category", "general")

                    # Validate category
                    if category in MEMORY_CATEGORIES:
                        logger.info(f"Categorized content as: {category} (sophisticated)")
                        return category
                    else:
                        logger.warning(f"Invalid category '{category}' from sophisticated extraction, defaulting to 'general'")
                        return "general"
                else:
                    logger.info("No valid items extracted, using 'general' category")
                    return "general"

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from sophisticated extraction: {e}")
                logger.error(f"Raw response: {response_text}")
                return "general"

        except Exception as e:
            logger.error(f"Error in sophisticated categorization: {e}")
            return "general"

    # DEPRECATED: Simple categorization method
    # This is kept for backward compatibility but should not be used for new code.
    # Use categorize_memory_sophisticated instead for consistent filtering and better quality.
    def categorize_memory(self, content: str) -> str:
        """Categorize memory content using sophisticated knowledge extraction (DEPRECATED: use categorize_memory_sophisticated)."""
        import warnings
        warnings.warn(
            "categorize_memory is deprecated. Use categorize_memory_sophisticated for better categorization.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.categorize_memory_sophisticated(content)

def categorize_content_sophisticated(content: str) -> str:
    """
    Categorize content using sophisticated knowledge extraction.
    This is the preferred method for categorization.

    Args:
        content: The content to categorize

    Returns:
        str: The best category found, or 'general' as fallback
    """
    try:
        # Create a temporary client for categorization
        client = GeminiClient()
        return client.categorize_memory_sophisticated(content)
    except Exception as e:
        logger.error(f"Error in categorize_content_sophisticated: {e}")
        return "general"

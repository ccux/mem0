"""
Configuration for Qdrant Memory Service
"""

import os
from typing import Optional

# Import centralized prompts
from prompts import (
    KNOWLEDGE_EXTRACTION_PROMPT,
    CATEGORIZATION_PROMPT,
    UPDATE_MEMORY_PROMPT,
    MEMORY_CATEGORIES,
    format_categorization_prompt,
    categorize_content_sophisticated
)

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))  # Changed default from 768 to 1536

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "memories")
QDRANT_USE_HTTPS = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"

# Memory service configuration
MEMORY_SERVICE_HOST = os.getenv("MEMORY_SERVICE_HOST", "0.0.0.0")
MEMORY_SERVICE_PORT = int(os.getenv("MEMORY_SERVICE_PORT", "8000"))

# API configuration (for main.py compatibility)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Knowledge extraction prompt - now using centralized version
KNOWLEDGE_EXTRACTION_PROMPT = KNOWLEDGE_EXTRACTION_PROMPT

# DEPRECATED: Categorization prompt - now using centralized version
# This is kept for backward compatibility but should not be used for new code.
# Use KNOWLEDGE_EXTRACTION_PROMPT instead for consistent filtering and better quality.
CATEGORIZATION_PROMPT = CATEGORIZATION_PROMPT

# Memory update prompt - now using centralized version
UPDATE_MEMORY_PROMPT = UPDATE_MEMORY_PROMPT

# DEPRECATED: Utility function to format categorization prompt
def format_categorization_prompt(content: str) -> str:
    """Format the categorization prompt with content (DEPRECATED)"""
    import warnings
    warnings.warn(
        "format_categorization_prompt is deprecated. Use KNOWLEDGE_EXTRACTION_PROMPT for sophisticated categorization.",
        DeprecationWarning,
        stacklevel=2
    )
    return format_categorization_prompt(content)

# Validation function
def is_valid_memory_category(category: str) -> bool:
    """Validate if a string is a valid memory category"""
    return category in MEMORY_CATEGORIES

def get_default_memory_category() -> str:
    """Get default memory category"""
    return "general"

def categorize_content_sophisticated(content: str) -> str:
    """
    Categorize content using sophisticated knowledge extraction.
    This is the preferred method for categorization.

    Args:
        content: The content to categorize

    Returns:
        str: The best category found, or 'general' as fallback
    """
    return categorize_content_sophisticated(content)

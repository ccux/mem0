import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "memories")

# Google AI Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

# Vector Configuration
VECTOR_DIMENSION = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # Use 1536 dimensions as supported by Gemini

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8002"))

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY environment variable is required")

# Memory Categories
MEMORY_CATEGORIES = [
    "person",     # Information about people or the user
    "work",       # Work, company or business related information
    "place",      # Information related to a Location or geography
    "event",      # Information about events or dates
    "task",       # Information about tasks or goals
    "general"     # Other important information that does not fit into the other categories
]

# LLM Categorization Prompt
CATEGORIZATION_PROMPT = """You are an expert at categorizing information. Analyze the following text and determine which category it belongs to.

Categories:
- person: Information about people or the user
- work: Work, company or business related information
- place: Information related to a Location or geography
- event: Information about events or dates
- task: Information about tasks or goals
- general: Other important information that does not fit into the other categories

Text to categorize: {content}

Respond with only the category name (person, work, place, event, task, or general). No explanation needed."""

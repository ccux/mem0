"""
Single Source of Truth for Knowledge Extraction Prompts (Python)

This module contains all knowledge extraction and categorization prompts
used across the Cognition Suite Python components. All other modules should
import from here to ensure consistency.
"""

# Memory categories used across the system
MEMORY_CATEGORIES = [
    'person',
    'work',
    'place',
    'event',
    'task',
    'general'
]

def is_valid_memory_category(category: str) -> bool:
    """Validate if a string is a valid memory category"""
    return category in MEMORY_CATEGORIES

def get_default_memory_category() -> str:
    """Get default memory category"""
    return "general"

# Sophisticated Knowledge Extraction Prompt
KNOWLEDGE_EXTRACTION_PROMPT = """
You are an expert at analyzing conversations and extracting valuable information, responsible for an AI memory database related to the user that is using the system.
You will be presented with user chat information, documents and other content that you are responsible for extracting important information from, and storing to a memory database that will provide better and more contextual responses in the future chat conversations.

Your task is to:
1. Extract ONLY key facts, preferences, commitments, or other important information that are relevant to remember
2. Format each piece of information as a complete, self-contained statement with context
3. Only extract factual information or clear user preferences, not opinions or hypotheticals
4. Assign each piece of information a category:
   - person: Information about people or the user
   - work: Work, company or business related information
   - place: Information related to a Location or geography
   - event: Information about events or dates
   - task: Information about tasks or goals
   - general: Other important information that does not fit into the other categories

**CRITICAL FILTERING RULES**:
- DO NOT extract test messages like "test", "hello", "hi", "this is a test", or similar trivial content
- DO NOT extract greetings, small talk, or casual conversation
- DO NOT extract requests for help without specific context (e.g., "Can you help me?" alone is not valuable)
- DO NOT extract hypothetical scenarios or examples unless they represent real user preferences
- DO NOT extract temporary or one-time questions that have no lasting value
- ONLY extract information that would genuinely help provide better responses in future conversations

**IMPORTANT**: If the input contains only images, page headers, minimal text, test content, greetings, or no meaningful/valuable content, return an empty array. Do NOT invent or hallucinate information that is not explicitly present in the input. Do not extract information that would not be helpful for future chat interactions.

Return a JSON array of extracted knowledge items, with each item having:
- content: The extracted knowledge in a complete sentence
- category: The assigned category
- confidence: A number from 0 to 1 indicating confidence level

Example:
User: "I need to finish the marketing report by Friday. Oh, and I'm moving to Seattle next month."
Output:
{
  "items": [
    {
      "content": "User needs to finish the marketing report by Friday",
      "category": "task",
      "confidence": 0.9
    },
    {
      "content": "User is moving to Seattle next month",
      "category": "place",
      "confidence": 0.85
    }
  ]
}

Example of trivial content to REJECT:
User: "This is a test"
Output:
{
  "items": []
}

Example of greeting to REJECT:
User: "Hello, how are you?"
Output:
{
  "items": []
}

Only include high-confidence information (>0.7) that has genuine value for future interactions. If there's nothing significant to extract, return an empty array.
"""

# DEPRECATED: Simple Categorization Prompt
# This is kept for backward compatibility but should not be used for new code.
# Use KNOWLEDGE_EXTRACTION_PROMPT instead for consistent filtering and better quality.
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

# Memory Update Prompt
UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "User is a software engineer"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Name is John",
                    "event" : "ADD"
                }
            ]
        }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it.
If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information.
Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is "Loves to play cricket with friends", then update the memory with the retrieved facts.
Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves cheese pizza", then you do not need to update it because they convey the same information.
If the direction is to update the memory, then you have to update it.
Please keep in mind while updating you have to keep the same ID.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "I really like cheese pizza"
            },
            {
                "id" : "1",
                "text" : "User is a software engineer"
            },
            {
                "id" : "2",
                "text" : "User likes to play cricket"
            }
        ]
    - Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Loves cheese and chicken pizza",
                    "event" : "UPDATE",
                    "old_memory" : "I really like cheese pizza"
                },
                {
                    "id" : "1",
                    "text" : "User is a software engineer",
                    "event" : "NONE"
                },
                {
                    "id" : "2",
                    "text" : "Loves to play cricket with friends",
                    "event" : "UPDATE",
                    "old_memory" : "User likes to play cricket"
                }
            ]
        }

3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it.
Please note to return the IDs in the output from the input IDs only and do not generate any new ID.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Dislikes cheese pizza"]
    - New Memory:
        {
        "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "DELETE"
                }
        ]
        }

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.
- **Example**:
    - Old Memory:
        [
            {
                "id" : "0",
                "text" : "Name is John"
            },
            {
                "id" : "1",
                "text" : "Loves cheese pizza"
            }
        ]
    - Retrieved facts: ["Name is John"]
    - New Memory:
        {
            "memory" : [
                {
                    "id" : "0",
                    "text" : "Name is John",
                    "event" : "NONE"
                },
                {
                    "id" : "1",
                    "text" : "Loves cheese pizza",
                    "event" : "NONE"
                }
            ]
        }
"""

def format_categorization_prompt(content: str) -> str:
    """Format the categorization prompt with content (DEPRECATED)"""
    import warnings
    warnings.warn(
        "format_categorization_prompt is deprecated. Use KNOWLEDGE_EXTRACTION_PROMPT for sophisticated categorization.",
        DeprecationWarning,
        stacklevel=2
    )
    return CATEGORIZATION_PROMPT.format(content=content)

def extract_knowledge_from_content(content: str) -> dict:
    """
    Extract knowledge from content using the sophisticated prompt.
    This is the preferred method for categorization.

    Args:
        content: The content to analyze

    Returns:
        dict: JSON response with extracted items
    """
    # This would integrate with the LLM in a real implementation
    # For now, return a mock response structure
    return {
        "items": [
            {
                "content": "Example extracted knowledge",
                "category": "general",
                "confidence": 0.8
            }
        ]
    }

def categorize_content_sophisticated(content: str) -> str:
    """
    Categorize content using sophisticated knowledge extraction.
    This replaces the simple categorization with the sophisticated approach.

    Args:
        content: The content to categorize

    Returns:
        str: The best category found, or 'general' as fallback
    """
    try:
        # Use sophisticated knowledge extraction
        result = extract_knowledge_from_content(content)

        if result.get("items") and len(result["items"]) > 0:
            # Get the highest confidence item
            best_item = max(result["items"], key=lambda x: x.get("confidence", 0))
            category = best_item.get("category", "general")

            # Validate category
            if category in MEMORY_CATEGORIES:
                return category
            else:
                return "general"
        else:
            return "general"

    except Exception as e:
        print(f"Error in sophisticated categorization: {e}")
        return "general"

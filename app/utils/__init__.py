"""
Utility functions and helpers.
"""

import logging
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    """
    Generate a hash for text content.
    Useful for caching embeddings.
    
    Args:
        text: Input text
        
    Returns:
        SHA-256 hash of the text
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Rough approximation: ~4 characters per token for English.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate a UUID string format.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        True if valid UUID format
    """
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_string))


def safe_json_loads(json_string: str, default: Optional[dict] = None) -> dict:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed dict or default
    """
    import json
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default or {}

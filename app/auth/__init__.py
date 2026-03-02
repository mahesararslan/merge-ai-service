"""Authentication module for AI service."""

from app.auth.api_key_auth import verify_api_key

__all__ = ["verify_api_key"]

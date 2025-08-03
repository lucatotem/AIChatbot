"""
Data Models

Pydantic models for request/response validation and API documentation.
These models ensure type safety and provide automatic API documentation
through FastAPI's integration with OpenAPI.

Author: Your Name
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Represents a user message sent to the chatbot.
    
    Attributes:
        message: The user's question or statement
        user_id: Optional identifier for conversation tracking
    """
    message: str = Field(..., description="The user's message to the chatbot", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier for session tracking")


class ChatResponse(BaseModel):
    """
    Represents the chatbot's response to a user message.
    
    Attributes:
        response: The generated response text
        sources: List of document sources used for the response
        confidence: Optional confidence score for the response quality
    """
    response: str = Field(..., description="The chatbot's response")
    sources: Optional[List[str]] = Field(None, description="Documents used to generate the response")
    confidence: Optional[float] = Field(None, description="Confidence score between 0 and 1")


class HealthResponse(BaseModel):
    """
    Health check response model for monitoring the application status.
    
    Attributes:
        status: Current health status of the application
        version: Application version number
        models_loaded: Whether AI models are loaded and ready
    """
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    version: str = Field(..., description="Current application version")
    models_loaded: bool = Field(..., description="Whether AI models are loaded and ready")

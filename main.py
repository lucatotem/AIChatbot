"""
Personal AI Assistant Backend

A FastAPI-based backend service that provides an intelligent chatbot interface
for personal websites and portfolios. Uses RAG (Retrieval-Augmented Generation)
to answer questions about professional background and experience.

Author: Your Name
Created: 2025
"""

import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from models import ChatMessage, ChatResponse, HealthResponse
from rag_system import initialize_rag_system
from chatbot import create_rag_chatbot

load_dotenv()

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for the application
rag_chatbot = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager that handles startup and shutdown events.
    Initializes the RAG system and loads ML models during startup.
    """
    global rag_chatbot, vector_store
    
    logger.info("Starting AI Chatbot Backend...")
    
    try:
        # Load configuration from environment variables
        documents_path = os.getenv("DOCUMENTS_PATH", "./documents")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        chat_model = os.getenv("CHAT_MODEL", "distilgpt2")
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Ensure documents directory exists
        os.makedirs(documents_path, exist_ok=True)
        
        # Initialize the RAG system with document processing
        logger.info("Initializing RAG system...")
        vector_store = initialize_rag_system(
            documents_path=documents_path,
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create the chatbot with RAG capabilities
        logger.info("Creating RAG chatbot...")
        rag_chatbot = create_rag_chatbot(chat_model, vector_store)
        
        logger.info("AI Chatbot Backend started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down AI Chatbot Backend...")

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Personal AI Assistant Backend",
    description="An intelligent chatbot backend for personal websites using RAG architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for web integration
cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint that provides basic information about the API.
    Useful for health checks and API discovery.
    """
    return {
        "message": "Personal AI Assistant Backend",
        "version": "1.0.0",
        "status": "running",
        "description": "RAG-powered chatbot for personal websites",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "documents": "/documents",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancer integration.
    Returns the current status of the application and its dependencies.
    """
    global rag_chatbot, vector_store
    
    models_loaded = (
        rag_chatbot is not None and 
        rag_chatbot.chatbot.is_loaded() and
        vector_store is not None
    )
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded=models_loaded
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint that processes user messages and returns AI responses.
    Uses RAG to provide contextually relevant answers based on uploaded documents.
    
    Args:
        message: ChatMessage containing the user's question and optional user_id
        
    Returns:
        ChatResponse with the AI's response, sources, and confidence score
    """
    global rag_chatbot
    
    if rag_chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Generate response using RAG system
        result = rag_chatbot.chat_with_context(
            user_message=message.message,
            user_id=message.user_id or "anonymous"
        )
        
        if not result.get("success", True):
            logger.warning(f"Chat generation failed: {result.get('error', 'Unknown error')}")
        
        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            confidence=result.get("confidence")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/documents")
async def list_documents():
    """
    Returns a list of all documents currently loaded in the knowledge base.
    Useful for debugging and understanding what information the bot has access to.
    """
    documents_path = os.getenv("DOCUMENTS_PATH", "./documents")
    docs_dir = Path(documents_path)
    
    if not docs_dir.exists():
        return {"documents": [], "count": 0}
    
    documents = []
    for file_path in docs_dir.rglob("*"):
        if file_path.is_file():
            documents.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "extension": file_path.suffix.lower()
            })
    
    return {
        "documents": documents,
        "count": len(documents)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load server configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

"""
Chatbot Core Module

Implements the main chatbot functionality using Hugging Face transformers
and RAG (Retrieval-Augmented Generation) for context-aware responses.
Handles model loading, response generation, and conversation management.

Author: Your Name
"""

import os
import logging
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class ChatBot:
    """
    Core chatbot class that handles AI model interactions and response generation.
    Supports multiple Hugging Face models with automatic fallback for reliability.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a specific language model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.chat_pipeline = None
        self.conversation_history = {}
        self.max_length = 512
    def load_model(self) -> None:
        """
        Load the specified Hugging Face model for text generation.
        Implements fallback strategy to ensure reliability even if primary model fails.
        """
        token = os.getenv("HUGGINGFACE_TOKEN")
        try:
            logger.info(f"Loading model: {self.model_name}")
            # Use different pipeline configurations based on model type
            if "DialoGPT" in self.model_name:
                self.chat_pipeline = pipeline(
                    "conversational",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1,  # Use CPU for compatibility
                    token=token
                )
            else:
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1,
                    max_length=self.max_length,
                    token=token
                )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a smaller, more reliable model
            try:
                logger.info("Falling back to distilgpt2...")
                self.model_name = "distilgpt2"
                self.chat_pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=-1,
                    max_length=self.max_length
                )
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")
                raise
    def generate_response(self, user_message: str, context: str = "", 
                         user_id: str = "default") -> Dict[str, Any]:
        """
        Generate a response to the user's message, optionally using provided context.
        
        Args:
            user_message: The user's input message
            context: Optional context from document retrieval
            user_id: User identifier for conversation tracking
            
        Returns:
            Dictionary containing response, model info, and success status
        """
        
        if self.chat_pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Construct prompt with optional context for better responses
            if context:
                prompt = f"Context: {context}\n\nUser: {user_message}\nAssistant:"
            else:
                prompt = f"User: {user_message}\nAssistant:"
            
            # Handle different model types with appropriate generation strategies
            if "DialoGPT" in self.model_name:
                from transformers import Conversation
                conversation = Conversation(user_message)
                result = self.chat_pipeline(conversation)
                response = result.generated_responses[-1]
            else:
                result = self.chat_pipeline(
                    prompt,
                    max_new_tokens=128,  # Limit the number of new tokens generated
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256,  # GPT-2 padding token
                    truncation=True
                )
                # Extract only the generated portion
                generated_text = result[0]['generated_text']
                response = generated_text[len(prompt):].strip()
                # Clean up response to prevent conversation bleeding
                if "User:" in response:
                    response = response.split("User:")[0].strip()
            
            return {
                "response": response,
                "model_used": self.model_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble generating a response right now. Please try again.",
                "model_used": self.model_name,
                "success": False,
                "error": str(e)
            }
    def is_loaded(self) -> bool:
        """Check if the model is properly loaded and ready for inference."""
        return self.chat_pipeline is not None


class RAGChatBot:
    """
    Enhanced chatbot that combines base chatbot functionality with 
    Retrieval-Augmented Generation (RAG) for more accurate, context-aware responses.
    """
    
    def __init__(self, chatbot: ChatBot, vector_store):
        """
        Initialize RAG chatbot with base chatbot and vector store.
        
        Args:
            chatbot: Base ChatBot instance
            vector_store: Vector database for document retrieval
        """
        self.chatbot = chatbot
        self.vector_store = vector_store
        
    def chat_with_context(self, user_message: str, k: int = 3, 
                         user_id: str = "default") -> Dict[str, Any]:
        """
        Generate a response using RAG - retrieves relevant context before generation.
        
        Args:
            user_message: User's input message
            k: Number of similar documents to retrieve
            user_id: User identifier for session tracking
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        
        try:
            # Retrieve relevant documents from vector store
            search_results = self.vector_store.similarity_search(user_message, k=k)
            
            # Process search results to extract context and sources
            context_parts = []
            sources = []
            
            for result in search_results:
                context_parts.append(result['content'])
                if 'source_file' in result['metadata']:
                    sources.append(result['metadata']['source_file'])
            
            # Combine retrieved context for prompt enhancement
            context = "\n\n".join(context_parts)
            
            # Generate response with enriched context
            response_data = self.chatbot.generate_response(
                user_message, 
                context=context, 
                user_id=user_id
            )
            
            # Add RAG-specific metadata to response
            response_data.update({
                "sources": list(set(sources)),  # Remove duplicate sources
                "context_used": bool(context_parts),
                "num_context_chunks": len(context_parts)
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            # Graceful fallback to regular chat without context
            return self.chatbot.generate_response(user_message, user_id=user_id)


def create_rag_chatbot(model_name: str, vector_store) -> RAGChatBot:
    """
    Factory function to create a fully configured RAG chatbot.
    
    Args:
        model_name: Hugging Face model identifier
        vector_store: Initialized vector store instance
        
    Returns:
        Ready-to-use RAGChatBot instance
    """
    chatbot = ChatBot(model_name)
    chatbot.load_model()
    return RAGChatBot(chatbot, vector_store)

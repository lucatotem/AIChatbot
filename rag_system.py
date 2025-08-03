"""
RAG System Implementation

Implements Retrieval-Augmented Generation (RAG) using LangChain and FAISS.
Handles document processing, embedding generation, and similarity search
to provide contextually relevant information for chatbot responses.

Author: Your Name
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles loading and processing of various document formats for RAG.
    Supports PDF, DOCX, and TXT files with intelligent text chunking.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor with chunking configuration.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    def load_documents(self, documents_path: str) -> List[Any]:
        """
        Load all supported documents from the specified directory.
        Automatically detects file types and uses appropriate loaders.
        
        Args:
            documents_path: Path to directory containing documents
            
        Returns:
            List of loaded document objects with metadata
        """
        documents = []
        docs_dir = Path(documents_path)
        
        if not docs_dir.exists():
            logger.warning(f"Documents directory {documents_path} does not exist")
            return documents
            
        for file_path in docs_dir.rglob("*"):
            if file_path.is_file():
                try:
                    # Select appropriate loader based on file extension
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() in ['.docx', '.doc']:
                        loader = Docx2txtLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path), encoding='utf-8')
                    else:
                        logger.info(f"Skipping unsupported file: {file_path}")
                        continue
                        
                    # Load document and add source metadata
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source_file'] = file_path.name
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        return documents
    def process_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks suitable for vector storage and retrieval.
        Maintains context through overlapping chunks.
        
        Args:
            documents: List of loaded document objects
            
        Returns:
            List of text chunks ready for embedding
        """
        if not documents:
            return []
            
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(texts)} chunks")
        return texts


class VectorStore:
    """
    Manages vector embeddings and similarity search using FAISS.
    Provides efficient storage and retrieval of document embeddings.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with embedding model.
        
        Args:
            embedding_model: Hugging Face model for generating embeddings
        """
        self.embedding_model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
    def create_vector_store(self, texts: List[Any]) -> None:
        """
        Create FAISS vector store from processed text chunks.
        Generates embeddings and builds searchable index.
        
        Args:
            texts: List of text chunks to embed and index
        """
        if not texts:
            logger.warning("No texts provided for vector store creation")
            return
            
        try:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Created vector store with {len(texts)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
            
    def save_vector_store(self, path: str) -> None:
        """
        Persist vector store to disk for faster subsequent loads.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
            
        try:
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            
    def load_vector_store(self, path: str) -> bool:
        """
        Load previously saved vector store from disk.
        
        Args:
            path: Directory path containing saved vector store
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if os.path.exists(path):
                self.vector_store = FAISS.load_local(
                    path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vector store loaded from {path}")
                return True
            else:
                logger.info(f"No existing vector store found at {path}")
                return False
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
            
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Find the most similar documents to a query using vector similarity.
        
        Args:
            query: Search query text
            k: Number of most similar documents to return
            
        Returns:
            List of documents with similarity scores and metadata
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
            
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []


def initialize_rag_system(documents_path: str, vector_store_path: str, 
                         embedding_model: str, chunk_size: int = 1000, 
                         chunk_overlap: int = 200) -> VectorStore:
    """
    Initialize the complete RAG system with document processing and vector storage.
    
    Args:
        documents_path: Path to directory containing source documents
        vector_store_path: Path for vector store persistence
        embedding_model: Hugging Face embedding model identifier
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks for context preservation
        
    Returns:
        Initialized and ready-to-use VectorStore instance
    """
    
    # Initialize processing components
    doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
    vector_store = VectorStore(embedding_model)
    
    # Try to load existing vector store for faster startup
    if vector_store.load_vector_store(vector_store_path):
        return vector_store
    
    # Process documents and create new vector store
    logger.info("Creating new vector store from documents...")
    documents = doc_processor.load_documents(documents_path)
    
    if not documents:
        logger.warning("No documents found. Creating empty vector store.")
        # Create placeholder document for testing
        from langchain.schema import Document
        test_doc = Document(
            page_content="This is a placeholder document. Please add your resume and other documents to the documents/ folder.",
            metadata={"source_file": "placeholder.txt"}
        )
        texts = [test_doc]
    else:
        texts = doc_processor.process_documents(documents)
    
    # Build and persist vector store
    vector_store.create_vector_store(texts)
    vector_store.save_vector_store(vector_store_path)
    
    return vector_store

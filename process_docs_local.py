#!/usr/bin/env python3

import logging
import os
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def process_documents_local():
    """
    Process documents locally and add them to the vector store.
    This is a simplified version of the Celery task for local development.
    """
    logger.info("Starting local document processing...")
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, 
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Create collection name
        collection_name_from_path = os.path.basename(
            os.path.normpath(settings.DOCS_PATH)
        )
        collection_name = f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"
        
        logger.info(f"Using collection name: {collection_name}")
        
        # Initialize vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        # Load documents
        logger.info(f"Loading documents from: {settings.DOCS_PATH}")
        loader = DirectoryLoader(
            settings.DOCS_PATH,
            glob="**/*.*",
            use_multithreading=True,
            show_progress=True,
        )
        documents = loader.load()
        
        if not documents:
            logger.info("No documents found to process.")
            return "No documents to process."
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        split_documents = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_documents)} document chunks")
        
        # Add documents to vector store
        logger.info("Adding document chunks to the vector store...")
        vector_store.add_documents(split_documents)
        
        logger.info("Successfully finished processing documents.")
        return f"Successfully processed {len(split_documents)} document chunks."
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    result = process_documents_local()
    print(result) 
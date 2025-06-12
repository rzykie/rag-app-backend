#!/usr/bin/env python3

import logging
import os
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

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
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, 
            base_url=settings.OLLAMA_BASE_URL
        )
        
        collection_name_from_path = os.path.basename(
            os.path.normpath(settings.DOCS_PATH)
        )
        collection_name = f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"
        
        logger.info(f"Using collection name: {collection_name}")
        
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        logger.info(f"Loading documents from: {settings.DOCS_PATH}")
        
        # Import OCR processing function
        from pdf_ocr_processor import process_pdf_with_ocr
        
        documents = []
        
        # Process PDF files with OCR
        pdf_path = os.path.join(settings.DOCS_PATH, "Company Manual.pdf")
        if os.path.exists(pdf_path):
            logger.info(f"Processing PDF with OCR: {pdf_path}")
            try:
                pdf_documents = process_pdf_with_ocr(pdf_path)
                logger.info(f"OCR processing extracted from {len(pdf_documents)} pages")
                documents.extend(pdf_documents)
            except Exception as e:
                logger.error(f"PDF OCR processing failed: {e}")
        
        # Load other text files
        loader = DirectoryLoader(
            settings.DOCS_PATH,
            glob="**/*.txt",  # Only text files, PDFs handled above
            use_multithreading=True,
            show_progress=True,
        )
        text_documents = loader.load()
        documents.extend(text_documents)
        
        if not documents:
            logger.info("No documents found to process.")
            return "No documents to process."
        
        logger.info(f"Loaded {len(documents)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        
        split_documents = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_documents)} document chunks")
        
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
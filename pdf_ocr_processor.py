#!/usr/bin/env python3

import logging
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings as ChromaSettings
from config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def convert_pdf_page_to_image(pdf_path, page_num, zoom=2.0):
    """
    Convert a PDF page to a PIL Image for OCR processing.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Create transformation matrix for higher resolution
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return image
        
    except Exception as e:
        logger.error(f"Error converting page {page_num + 1} to image: {e}")
        return None

def extract_text_from_image(image, page_num):
    """
    Extract text from an image using OCR.
    """
    try:
        # Simple OCR extraction
        text = pytesseract.image_to_string(image, lang='eng')
        
        # Clean up the text
        cleaned_text = text.strip()
        
        logger.info(f"Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
        
        if cleaned_text:
            # Show sample of extracted text
            sample = cleaned_text[:200].replace('\n', ' ')
            logger.info(f"  Sample: {sample}...")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error extracting text from page {page_num + 1}: {e}")
        return ""

def process_pdf_with_ocr(pdf_path):
    """
    Process entire PDF using page-by-page OCR extraction.
    """
    try:
        logger.info(f"Starting OCR processing of: {pdf_path}")
        
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"PDF has {total_pages} pages")
        
        documents = []
        total_chars_extracted = 0
        
        for page_num in range(total_pages):
            logger.info(f"Processing page {page_num + 1}/{total_pages}...")
            
            image = convert_pdf_page_to_image(pdf_path, page_num)
            
            if image is None:
                logger.warning(f"Failed to convert page {page_num + 1} to image")
                continue
            
            # Extract text using OCR
            extracted_text = extract_text_from_image(image, page_num)
            
            if extracted_text.strip():
                doc = Document(
                    page_content=extracted_text,
                    metadata={
                        'source': pdf_path,
                        'page': page_num + 1,
                        'extraction_method': 'ocr',
                        'text_length': len(extracted_text)
                    }
                )
                documents.append(doc)
                total_chars_extracted += len(extracted_text)
            else:
                logger.warning(f"No text extracted from page {page_num + 1}")
        
        logger.info(f"=== OCR EXTRACTION COMPLETE ===")
        logger.info(f"Processed {total_pages} pages")
        logger.info(f"Successfully extracted from {len(documents)} pages")
        logger.info(f"Total characters extracted: {total_chars_extracted:,}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error during PDF OCR processing: {e}", exc_info=True)
        return []

def search_extracted_content(documents, search_terms):
    """
    Search for specific terms in the extracted content.
    """
    logger.info("=== SEARCHING EXTRACTED CONTENT ===")
    
    # Combine all text for searching
    full_text = " ".join(doc.page_content for doc in documents).lower()
    
    for term in search_terms:
        count = full_text.count(term.lower())
        logger.info(f"'{term}': {count} occurrences")
        
        if count > 0:
            # Find and show context
            term_lower = term.lower()
            idx = full_text.find(term_lower)
            start = max(0, idx - 150)
            end = min(len(full_text), idx + 300)
            context = full_text[start:end].replace('\n', ' ').strip()
            logger.info(f"  Context: ...{context}...")

def process_and_index_pdf():
    """
    Main function to process PDF with OCR and index the results.
    """
    try:
        pdf_path = os.path.join(settings.DOCS_PATH, "Company Manual.pdf")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found at: {pdf_path}")
            return
        
        logger.info(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")
        
        # Process PDF with OCR
        documents = process_pdf_with_ocr(pdf_path)
        
        if not documents:
            logger.error("No content extracted from PDF!")
            return
        
        # Search for key terms
        search_terms = ['mugna', 'core value', 'values', 'mission', 'vision', 'principle']
        search_extracted_content(documents, search_terms)
        
        # Split documents into chunks
        logger.info("=== SPLITTING INTO CHUNKS ===")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        
        split_documents = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_documents)} chunks")
        
        # Index the documents
        logger.info("=== INDEXING DOCUMENTS ===")
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, 
            base_url=settings.OLLAMA_BASE_URL
        )
        
        collection_name_from_path = os.path.basename(
            os.path.normpath(settings.DOCS_PATH)
        )
        collection_name = f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"
        
        # Create vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        # Delete existing collection to start fresh
        try:
            vector_store.delete_collection()
            logger.info("Deleted existing collection")
        except:
            pass
        
        # Create new collection and add documents
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        logger.info("Adding document chunks to vector store...")
        vector_store.add_documents(split_documents)
        
        logger.info(f"Successfully processed and indexed {len(split_documents)} chunks")
        
        # Test search
        logger.info("=== TESTING SEARCH ===")
        test_queries = [
            "core values of Mugna",
            "Mugna values", 
            "company values",
            "mission",
            "principles"
        ]
        
        for query in test_queries:
            docs = vector_store.similarity_search(query, k=3)
            logger.info(f"Query '{query}' returned {len(docs)} results")
            for i, doc in enumerate(docs):
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"  Result {i+1}: {preview}...")
        
        return f"Successfully processed {len(split_documents)} chunks using OCR"
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    result = process_and_index_pdf()
    print(result) 
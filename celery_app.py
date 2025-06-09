import logging
import os
from celery import Celery
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from urllib.request import URLError

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Celery config from environment variables, with defaults for local development
celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Initialize Celery
celery = Celery(__name__, broker=celery_broker_url, backend=celery_result_backend)


@celery.task(
    name="process_documents",
    bind=True,
    autoretry_for=(
        ConnectionError,
        URLError,
    ),
    retry_kwargs={"max_retries": 5, "countdown": 30},
)
def process_documents(self):
    """
    Celery task to load documents, create embeddings, and add them to the vector store.
    This task will automatically retry on connection errors.
    """
    logger.info("Starting document processing task...")
    try:
        embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
        )

        collection_name_from_path = os.path.basename(
            os.path.normpath(settings.DOCS_PATH)
        )
        collection_name = (
            f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"
        )

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )

        loader = DirectoryLoader(
            settings.DOCS_PATH,
            glob="**/*.*",
            use_multithreading=True,
            show_progress=True,
        )
        documents = loader.load_and_split()

        if not documents:
            logger.info("No documents found to process.")
            return "No documents to process."

        logger.info(f"Adding {len(documents)} document chunks to the vector store.")
        vector_store.add_documents(documents)
        logger.info("Successfully finished processing documents.")
        return f"Successfully processed {len(documents)} document chunks."
    except (ConnectionError, URLError) as e:
        logger.warning(f"Connection to Ollama failed: {e}. Retrying in 30 seconds...")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during document processing: {e}",
            exc_info=True,
        )
        # Do not retry for unexpected errors
        raise

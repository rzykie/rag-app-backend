from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables.
    """

    # Load settings from a .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Application settings
    DOCS_PATH: str = "documents"
    COLLECTION_NAME_PREFIX: str = "rag-collection-"

    # LLM and Embedding Model Settings
    LANGUAGE_MODEL: str = "qwen3:0.6b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"

    # ChromaDB Settings
    PERSIST_DIRECTORY: str = "./chroma_langchain_db"


# Create a single settings instance to be used throughout the application
settings = Settings()

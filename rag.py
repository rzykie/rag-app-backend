import logging
import os

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings

from config import settings


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LangChainRAG:
    def __init__(self):
        """Initialize LangChain RAG application using settings from config."""

        # Use settings from the config module
        self.docs_path = settings.DOCS_PATH
        self.language_model = settings.LANGUAGE_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        self.persist_directory = settings.PERSIST_DIRECTORY
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        # Create a unique collection name from the docs path
        collection_name_from_path = os.path.basename(os.path.normpath(self.docs_path))
        self.collection_name = (
            f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"
        )

        # Initialize components
        self.llm = self._initialize_llm()
        self.documents = self._load_documents()
        self.embeddings = self._initialize_embeddings()
        self.retriever = self._create_retriever()
        self.rag_chain = self._create_rag_chain()

    def _initialize_llm(self):
        """Initialize the language model"""
        logger.info(f"Initializing LLM: {self.language_model}")

        return ChatOllama(
            model=self.language_model,
            base_url=self.ollama_base_url,
            extract_reasoning=True,
            num_predict=-1,
        )

    def _load_documents(self):
        """Load documents from a directory"""
        logger.info(f"Loading documents from {self.docs_path}")

        loader = DirectoryLoader(self.docs_path, glob="**/*.txt", loader_cls=TextLoader)
        return loader.load_and_split()

    def _initialize_embeddings(self):
        """Initialize embeddings"""
        logger.info(f"Initializing embeddings: {self.embedding_model}")

        return OllamaEmbeddings(
            model=self.embedding_model, base_url=self.ollama_base_url
        )

    def _create_retriever(self):
        """Create the vector store and retriever"""
        logger.info(
            f"Creating ChromaDB vector store with collection: {self.collection_name}"
        )

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )

        if not vector_store.get()["documents"]:
            logger.info("No documents found in collection, adding new ones.")
            vector_store.add_documents(self.documents)
        else:
            logger.info("Existing documents found in collection, skipping add.")

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 50,
                "fetch_k": 50,
                "lambda_mult": 1,
            },
        )

        return retriever

    def _create_rag_chain(self):
        """Create the RAG chain with prompt template"""

        template = """
        You are a friendly and professional HR assistant for our company.
        Your goal is to help employees by answering their questions based on the official company handbook.

        Instructions:
        1.  Answer the user's question using ONLY the context provided below.
        2.  Be clear and concise in your response.
        3.  If the information is a list (e.g., types of leave), use bullet points for readability.
        4.  If the answer cannot be found in the provided context, respond with: "I'm sorry, but I couldn't find any information about that in the company handbook. Please reach out to HR for more details."
        5.  Do not make up answers or provide information from outside the context.

        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("human", "{input}")],
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        return create_retrieval_chain(self.retriever, question_answer_chain)

    def generate_response(self, query):
        """Generate response from the RAG chain"""
        logger.info(f"Generating response for query: {query}")

        try:
            response = self.rag_chain.invoke({"input": query})
            return response["answer"]
        except Exception as e:
            logger.error(f"Error response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

import logging

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.text import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LangChainRAG:
    def __init__(
        self, file_path, language_model="qwen3:0.6b", embedding_model="nomic-embed-text"
    ):
        """Initialize LangChain RAG application"""

        self.file_path = file_path
        self.language_model = language_model
        self.embedding_model = embedding_model

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
            model=self.language_model, extract_reasoning=True, num_predict=-1
        )

    def _load_documents(self):
        """Load documents"""
        logger.info("Loading documents")

        loader = TextLoader(self.file_path)
        return loader.load_and_split()

    def _initialize_embeddings(self):
        """Initialize embeddings"""
        logger.info(f"Initializing embeddings: {self.embedding_model}")

        return OllamaEmbeddings(model=self.embedding_model)

    def _create_retriever(self):
        """Create the vector store and retriever"""
        logger.info("Creating ChromaDB vector store")

        vector_store = Chroma(
            collection_name="langchain_rag_data",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
            client_settings=Settings(anonymized_telemetry=False),
        )

        if not vector_store.get()["documents"]:
            vector_store.add_documents(self.documents)

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
        You are a chatbot that answers questions about the context provided.

        Instructions:
        Answer the question using only the provided context. Respond in paragraph form only.

        Context: {context}
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


def main():
    file_path = "main.py"

    rag_app = LangChainRAG(file_path)

    query = "What is the language model used?"

    result = rag_app.generate_response(query)

    print(result)


if __name__ == "__main__":
    main()

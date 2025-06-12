import logging
import os

from chromadb.config import Settings as ChromaSettings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import settings


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LangChainRAG:
    def __init__(self):
        """
        Initializes the RAG application's core components for querying.
        Document loading and processing is handled separately.
        """
        self.language_model = settings.LANGUAGE_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.retriever = self._create_retriever()
        self.rag_chain = self._create_rag_chain()

    def _initialize_llm(self):
        """Initialize the Ollama language model"""
        return ChatOllama(
            model=self.language_model,
            base_url=self.ollama_base_url,
            temperature=0.1,  # Slightly higher for more natural responses
            top_p=0.9,        # Nucleus sampling for better quality
            repeat_penalty=1.1, # Reduce repetition
        )

    def _initialize_embeddings(self):
        """Initialize Ollama embeddings"""
        return OllamaEmbeddings(
            model=self.embedding_model, base_url=self.ollama_base_url
        )

    def _create_retriever(self):
        """
        Creates a retriever that connects to the ChromaDB vector store.
        """
        collection_name = self._get_collection_name()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        return vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={
                "k": 10,       # Increase to get more results
                "fetch_k": 40, # Cast a wider net
                "lambda_mult": 0.5  # More diversity to capture different phrasings
            }
        )

    def _create_rag_chain(self):
        """Create the RAG chain with prompt template"""
        template = """
        You are a friendly and professional HR assistant for our company.
        Your goal is to help employees by answering their questions based on the official company handbook.

        Instructions:
        1. Answer the user's question using ONLY the context provided below.
        2. Be clear, concise, and professional in your response.
        3. Use bullet points for lists and organize information logically.
        4. If you need to reason through the context, you may optionally include your thinking process in <think></think> tags before your main response.
        5. If the answer cannot be found in the provided context, respond with: "I'm sorry, but I couldn't find any information about that in the company handbook. Please reach out to HR for more details."
        6. Do not make up answers or provide information from outside the context.
        7. Provide a direct, helpful answer after any thinking process.

        Context:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def _get_collection_name(self):
        """Creates a unique collection name from the docs path."""
        collection_name_from_path = os.path.basename(
            os.path.normpath(settings.DOCS_PATH)
        )
        return f"{settings.COLLECTION_NAME_PREFIX}{collection_name_from_path}"

    def generate_response(self, query):
        """Generate response from the RAG chain"""
        logger.info(f"Generating response for query: {query}")
        try:
            response = self.rag_chain.invoke({"input": query})
            raw_answer = response["answer"]
            
            # Parse and separate thinking section if present
            if "<think>" in raw_answer and "</think>" in raw_answer:
                # Extract thinking section
                think_start = raw_answer.find("<think>") + 7
                think_end = raw_answer.find("</think>")
                thinking_process = raw_answer[think_start:think_end].strip()
                
                # Extract main response (everything after </think>)
                main_response = raw_answer[think_end + 8:].strip()
                
                # Structure the response
                structured_response = {
                    "thinking": thinking_process,
                    "answer": main_response
                }
                return structured_response
            else:
                # No thinking section, return as normal
                return {"answer": raw_answer}
                
        except Exception as e:
            logger.error(f"Error response: {str(e)}")
            return {"error": f"Sorry, I encountered an error: {str(e)}"}

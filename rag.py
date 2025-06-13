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
                "k": 5,        # Reduce to get more focused results
                "fetch_k": 20, # Reduce initial fetch
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

    def _create_rag_chain(self):
        """Create the RAG chain with prompt template"""
        template = """
        <system_prompt>
YOU ARE AN ENTHUSIASTIC AND PROFESSIONAL AI HR ASSISTANT FOR A FAST-GROWING TECH STARTUP. YOUR PRIMARY MISSION IS TO **UNDERSTAND**, **SUMMARIZE**, AND **CONTEXTUALIZE** THE COMPANY MANUAL TO HELP NEW AND EXISTING EMPLOYEES QUICKLY ORIENT THEMSELVES WITH COMPANY POLICIES, VALUES, AND CULTURE.

###OBJECTIVES###

- YOU MUST **READ AND INTERNALIZE** THE COMPANY MANUAL (PROVIDED AS CONTEXT)
- **GENERATE ENTHUSIASTIC, PROFESSIONAL SUMMARIES** OF THE CONTENT FOR EMPLOYEES TO EASILY UNDERSTAND
- **TRANSLATE POLICY LANGUAGE** INTO FRIENDLY, CLEAR, AND ACTIONABLE INSIGHTS
- **EMPHASIZE CULTURE, VALUES, AND EXPECTATIONS** IN A WAY THAT BUILDS TEAM SPIRIT
- YOU SHOULD **PROACTIVELY ANTICIPATE EMPLOYEE QUESTIONS** AND ANSWER THEM BASED ON THE MANUAL
- MAINTAIN AN OPTIMISTIC, SUPPORTIVE, AND EXPERT TONE AT ALL TIMES

###CHAIN OF THOUGHTS###

FOLLOW THESE STEPS TO CREATE AN EFFECTIVE SUMMARY RESPONSE:

1. **UNDERSTAND**:
   - THOROUGHLY REVIEW THE COMPANY MANUAL AND IDENTIFY CORE SECTIONS (e.g., Mission, Values, Code of Conduct, Benefits, Time Off, Communication Norms, Onboarding)

2. **BASICS**:
   - RECOGNIZE THE STARTUP ENVIRONMENT: FAST-PACED, INNOVATIVE, TEAM-DRIVEN
   - IDENTIFY LANGUAGE THAT MAY BE TOO FORMAL OR TECHNICAL FOR EVERYDAY USE

3. **BREAK DOWN**:
   - SPLIT THE MANUAL INTO DIGESTIBLE SEGMENTS: COMPANY CULTURE, DAILY WORKFLOW, POLICIES, EMPLOYEE BENEFITS, ETC.
   - CREATE ONE SUMMARY PER SECTION IF REQUESTED

4. **ANALYZE**:
   - ISOLATE KEY TERMS (e.g., “flexible hours,” “remote-first,” “unlimited PTO”) AND CLARIFY THEIR MEANING
   - MAP POLICIES TO REAL-WORLD SCENARIOS EMPLOYEES MIGHT ENCOUNTER

5. **BUILD**:
   - CRAFT SUMMARIES THAT ARE FRIENDLY, CLEAR, AND ENCOURAGING
   - TIE EVERYTHING BACK TO THE COMPANY’S CORE VALUES (e.g., innovation, ownership, transparency)

6. **EDGE CASES**:
   - FLAG AMBIGUITIES OR INCONSISTENCIES THAT EMPLOYEES MAY MISUNDERSTAND
   - PROACTIVELY EXPLAIN COMMONLY CONFUSED POLICIES (e.g., leave policy vs. holiday schedule)

7. **FINAL ANSWER**:
   - RESPOND WITH A POLISHED, PROFESSIONAL, YET WARM SUMMARY THAT SHOWS YOU’RE A RELIABLE PARTNER IN HR

###WHAT NOT TO DO###

- DO NOT COPY MANUAL SECTIONS VERBATIM WITHOUT SUMMARIZING OR CLARIFYING
- NEVER USE A COLD, CORPORATE, OR ROBOTIC TONE — AVOID BEING TOO FORMAL
- DO NOT IGNORE QUESTIONS THAT FALL UNDER THE MANUAL'S SCOPE
- NEVER GIVE GENERIC OR VAGUE ANSWERS — ALWAYS BE SPECIFIC AND CONTEXTUAL
- AVOID MISREPRESENTING POLICIES OR GIVING UNVERIFIED ADVICE
- DO NOT OMIT MENTION OF THE COMPANY'S CORE VALUES OR CULTURE IN SUMMARIES

###FEW-SHOT EXAMPLES###

**USER INPUT**: "Can you explain the vacation policy?"

**ASSISTANT OUTPUT**:
"Absolutely! 🌴 At our company, we believe in balance. That’s why we offer **unlimited paid time off** — yes, unlimited! 🎉 As long as your work is covered and your team is in the loop, you're encouraged to take the rest you need. We trust our team to make responsible decisions and support one another. Just give your manager a heads-up in advance, and enjoy your well-deserved time off!"

---

**USER INPUT**: "What are the company’s core values?"

**ASSISTANT OUTPUT**:
"Great question! 🚀 Our core values drive everything we do:
- **Ownership**: We take initiative and follow through.
- **Transparency**: We communicate openly and honestly.
- **Velocity**: We move fast and iterate faster.
- **Empathy**: We listen, support, and uplift each other.

These aren’t just buzzwords — they’re how we work, grow, and thrive together."

</system_prompt>

        Here is the relevant information to help answer the question:
        {context}

        Question: {input}
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

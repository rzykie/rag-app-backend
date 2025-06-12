import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from rag import LangChainRAG

app = FastAPI(
    title="Company Policy RAG API",
    description="A RAG-based API to answer questions about company policies, with automated data ingestion.",
    version="3.0.0",
)

# This initializes the RAG application components without loading documents.
# The application will start instantly.
rag_app = LangChainRAG()


class QueryRequest(BaseModel):
    query: str


@app.get("/health", summary="Check service health and connection to Ollama")
async def health_check():
    """
    Performs a health check on the API and its connection to the Ollama service.
    Returns a 200 OK if the connection is successful, otherwise a 503 Service Unavailable.
    """
    try:
        # Use a short timeout to avoid long waits
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
            return {"api_status": "up", "ollama_status": "up"}
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise HTTPException(
            status_code=503,
            detail={"api_status": "up", "ollama_status": "down", "error": str(e)},
        )


@app.post("/query", summary="Get answers about company policy")
async def query(request: QueryRequest):
    """
    Accepts a query and returns a response from the RAG model.
    The document ingestion process runs automatically in the background.
    """
    response = rag_app.generate_response(request.query)
    
    # Handle structured response with thinking section
    if isinstance(response, dict):
        if "thinking" in response and "answer" in response:
            return {
                "thinking": response["thinking"],
                "response": response["answer"]
            }
        elif "answer" in response:
            return {"response": response["answer"]}
        elif "error" in response:
            return {"response": response["error"]}
    
    # Fallback for string responses
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

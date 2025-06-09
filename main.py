from fastapi import FastAPI
from pydantic import BaseModel
from rag import LangChainRAG

app = FastAPI(
    title="Company Policy RAG API",
    description="A RAG-based API to answer questions about company policies.",
    version="1.0.0",
)

# Initialize the RAG application
rag_app = LangChainRAG()


class QueryRequest(BaseModel):
    query: str


@app.post("/query", summary="Get answers about company policy")
async def query(request: QueryRequest):
    """
    Accepts a query in a JSON object and returns a response from the RAG model based on the company handbook.

    - **query**: The question you want to ask about the company policy.
    """
    response = rag_app.generate_response(request.query)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

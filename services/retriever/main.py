from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/search")
async def search(request: SearchRequest):
    # Mock Retrieval Logic
    # Later, this will query pgvector
    mock_docs = [
        f"Doc 1 relevant to {request.query}",
        f"Doc 2 relevant to {request.query}",
        f"Doc 3 relevant to {request.query}"
    ]
    return {"documents": mock_docs}
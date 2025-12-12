import os
from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
from openai import OpenAI  

app = FastAPI()

# Configuration
LLM_BASE_URL = os.getenv("LLM_API_BASE", "http://host.docker.internal:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "llama3")

# Initialize Client
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="ollama" 
)

class QueryRequest(BaseModel):
    query: str
    topology: str = "sequential"
    defense: str = "off"

@app.post("/chat")
async def chat(request: QueryRequest):
    # 1. Topology Routing
    if request.topology == "sequential":
        return await run_sequential_rag(request.query)
    else:
        raise HTTPException(status_code=400, detail="Topology not implemented")

async def run_sequential_rag(query):
    # Step A: Retrieve (Still Mock for now, but integration ready)
    async with httpx.AsyncClient() as http_client:
        retrieval_resp = await http_client.post(
            "http://retriever:8001/search", 
            json={"query": query, "k": 3}
        )
        documents = retrieval_resp.json()["documents"]

    # Step B: Construct Prompt
    context_text = "\n".join(documents)
    
    # Secure RAG Prompt Template
    system_prompt = "You are a helpful AI assistant. Use the provided context to answer the question."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    # Step C: Generate (Real Call to Ollama)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Determinism for research
            seed=42
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"LLM Error: {str(e)}"
    
    return {
        "response": answer, 
        "context_used": documents,
        "model": LLM_MODEL
    }
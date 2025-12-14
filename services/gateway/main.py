import os
import logging
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

app = FastAPI(title="Secure RAG Gateway")

# Configuration
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://host.docker.internal:11434/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "secure-rag-llama3")

class ChatRequest(BaseModel):
    query: str
    topology: str = "sequential"  # Default to standard RAG
    profile: str = "P1"
    seed: int = 42

class ChatResponse(BaseModel):
    response: str
    context_used: List[str]
    model: str

@app.get("/health")
def health_check():
    return {"status": "ok", "llm": LLM_MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    logger.info(f"Received query: '{request.query}' [Topology: {request.topology}]")
    
    try:
        # 1. RETRIEVAL PHASE
        # Call the Retriever Service (Internal Docker Network)
        logger.info(f"Calling Retriever at {RETRIEVER_URL}...")
        try:
            ret_resp = requests.post(
                f"{RETRIEVER_URL}/search",
                json={"query": request.query, "k": 3, "profile": request.profile},
                timeout=5
            )
            ret_resp.raise_for_status()
            retrieved_docs = ret_resp.json().get("documents", [])
            doc_ids = [d['id'] for d in retrieved_docs]
            logger.info(f"Retrieved {len(retrieved_docs)} docs: {doc_ids}")
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            # Fail open or closed? For research, we fail loudly.
            raise HTTPException(status_code=503, detail=f"Retriever unavailable: {str(e)}")

        # 2. AUGMENTATION PHASE
        # Construct the context block
        context_text = "\n\n".join([
            f"[Document {d['id']}]: {d['content']}" 
            for d in retrieved_docs
        ])
        
        system_prompt = (
            "You are a secure, helpful assistant. Answer the user query based ONLY on the context provided below.\n"
            "If the answer is not in the context, say 'I do not know'.\n"
            "Do not use outside knowledge.\n\n"
            f"Context:\n{context_text}"
        )

        # 3. GENERATION PHASE
        # Call the Local LLM (Ollama)
        logger.info(f"Calling LLM ({LLM_MODEL_NAME}) at {LLM_API_BASE}...")
        
        llm_payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query}
            ],
            "stream": False,
            "temperature": 0.0, # Deterministic
            "seed": request.seed
        }
        
        try:
            llm_resp = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                json=llm_payload,
                timeout=60 
            )
            llm_resp.raise_for_status()
            llm_data = llm_resp.json()
            generated_text = llm_data['choices'][0]['message']['content']
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to LLM at {LLM_API_BASE}")
            raise HTTPException(status_code=503, detail="LLM Service Unavailable. Check LLM_API_BASE configuration.")
            
        return ChatResponse(
            response=generated_text,
            context_used=doc_ids,
            model=LLM_MODEL_NAME
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Gateway Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
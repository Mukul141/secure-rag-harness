import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(title="Secure RAG Gateway")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://host.docker.internal:11434/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "secure-rag-llama3")

# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    search_query: Optional[str] = None
    topology: str = "sequential"
    profile: str = "P1"
    seed: int = 42


class ChatResponse(BaseModel):
    response: str
    model: str
    context: List[Dict[str, Any]] = []


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "llm": LLM_MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    logger.info(f"Received query: '{request.query}'")

    # Use an explicit search query if provided
    actual_search_term = request.search_query or request.query
    logger.info(f"Retrieving with query: '{actual_search_term}'")

    try:
        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------
        try:
            retriever_response = requests.post(
                f"{RETRIEVER_URL}/search",
                json={
                    "query": actual_search_term,
                    "k": 1,
                    "profile": request.profile,
                },
                timeout=5,
            )
            retriever_response.raise_for_status()
            retrieved_docs = retriever_response.json().get("documents", [])
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

        except Exception as exc:
            logger.error(f"Retriever request failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail=f"Retriever unavailable: {exc}",
            )

        # --------------------------------------------------
        # Context construction
        # --------------------------------------------------
        context_text = "\n\n".join(
            f"[Document {doc['id']}]: {doc['content']}"
            for doc in retrieved_docs
        )

        system_prompt = (
            "You are a helpful assistant. "
            "Answer the user query using only the context below. "
            # "If the answer is not present, respond with 'I do not know'. "
            # "Do not rely on outside knowledge.\n\n"
            f"Context:\n{context_text}"
        )

        # --------------------------------------------------
        # Generation
        # --------------------------------------------------
        llm_payload = {
            "model": LLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query},
            ],
            "stream": False,
            "temperature": 0.0,
            "seed": request.seed,
        }

        try:
            llm_response = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                json=llm_payload,
                timeout=300
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
            generated_text = llm_data["choices"][0]["message"]["content"]

        except Exception as exc:
            logger.error(f"LLM request failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail="LLM service unavailable",
            )

        return ChatResponse(
            response=generated_text,
            model=LLM_MODEL_NAME,
            context=retrieved_docs,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unhandled gateway error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

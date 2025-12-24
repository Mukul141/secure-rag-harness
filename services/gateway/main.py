import os
import logging
import requests
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import middleware
from middleware import check_policy, log_telemetry

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")
app = FastAPI(title="Secure RAG Gateway")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://retriever:8001")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://host.docker.internal:11434/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "secure-rag-llama3")

# ------------------------------------------------------------------
# Template loading
# ------------------------------------------------------------------

TEMPLATES = {}
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def load_templates():
    """
    Loads system prompt templates from disk.
    Falls back to defaults if templates are missing.
    """
    try:
        if not os.path.exists(TEMPLATE_DIR):
            os.makedirs(TEMPLATE_DIR)

        def read_template(name, default):
            path = os.path.join(TEMPLATE_DIR, f"{name}.txt")
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            return default

        TEMPLATES["vanilla"] = read_template(
            "vanilla",
            "Context:\n{context_text}",
        )
        TEMPLATES["skeptical"] = read_template(
            "skeptical",
            "Context:\n{context_text}",
        )

        logger.info("System prompt templates loaded.")

    except Exception as exc:
        logger.error(f"Failed to load templates: {exc}")
        TEMPLATES["vanilla"] = "Context:\n{context_text}"


@app.on_event("startup")
async def startup_event():
    load_templates()

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
async def chat_handler(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    logger.info(f"Received query: '{request.query}'")

    actual_search_term = request.search_query or request.query

    # Retrieval
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
        raise HTTPException(status_code=503, detail="Retriever unavailable")

    # Policy checks on query and retrieved context
    check_policy(request.query, retrieved_docs)

    # Context construction and templating
    context_text = "\n\n".join(
        f"[Document {doc['id']}]: {doc['content']}"
        for doc in retrieved_docs
    )

    template_key = "skeptical" if request.profile == "P2" else "vanilla"
    system_prompt = TEMPLATES.get(
        template_key, TEMPLATES["vanilla"]
    ).format(context_text=context_text)

    # LLM generation
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
            timeout=90,
        )
        llm_response.raise_for_status()
        generated_text = llm_response.json()["choices"][0]["message"]["content"]

    except Exception as exc:
        logger.error(f"LLM request failed: {exc}")
        raise HTTPException(status_code=503, detail="LLM unavailable")

    # Telemetry logging (async)
    latency = time.time() - start_time
    background_tasks.add_task(
        log_telemetry,
        {
            "timestamp": time.time(),
            "latency": latency,
            "profile": request.profile,
            "docs_retrieved": len(retrieved_docs),
            "status": "success",
        },
    )

    return ChatResponse(
        response=generated_text,
        model=LLM_MODEL_NAME,
        context=retrieved_docs,
    )

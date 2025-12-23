import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "vector_db")
DB_NAME = os.getenv("POSTGRES_DB", "ragdb")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

# ------------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------------

logger.info("Loading sentence transformer model from local path...")
model = SentenceTransformer("./model_data", device="cpu")
logger.info("Embedding model loaded.")

# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]


class IngestRequest(BaseModel):
    documents: List[Document]

# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------

def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
        )
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        raise


@app.on_event("startup")
def startup_db():
    """
    Initializes the database schema and required extensions.
    """
    try:
        conn = get_db_connection()
        conn.autocommit = True

        register_vector(conn)

        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(384)
            )
            """
        )

        cur.close()
        conn.close()
        logger.info("Database schema initialized.")

    except Exception as exc:
        logger.critical(f"Startup initialization failed: {exc}")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    logger.info(f"Received ingestion request for {len(request.documents)} documents.")

    conn = get_db_connection()
    register_vector(conn)
    cur = conn.cursor()

    try:
        indexed_count = 0

        for doc in request.documents:
            logger.info(f"Indexing document: {doc.id}")

            # Generate embedding
            embedding = model.encode(doc.text).tolist()

            # Serialize metadata
            metadata_json = json.dumps(doc.metadata)

            # Insert or update document
            cur.execute(
                """
                INSERT INTO documents (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (doc.id, doc.text, metadata_json, embedding),
            )

            indexed_count += 1

        conn.commit()
        logger.info(f"Successfully indexed {indexed_count} documents.")
        return {"status": "success", "indexed": indexed_count}

    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        cur.close()
        conn.close()


@app.post("/reset")
def reset_database():
    """
    Clears all stored documents.
    Used to ensure a clean state between experiments.
    """
    logger.warning("Reset request received. Truncating documents table.")

    conn = None
    try:
        conn = get_db_connection()
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE documents;")

        cur.close()
        conn.close()
        logger.info("Database reset completed.")
        return {"status": "success", "message": "Vector database truncated"}

    except Exception as exc:
        logger.error(f"Database reset failed: {exc}")
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)

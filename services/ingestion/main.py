import os
import json  
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import psycopg2
from pgvector.psycopg2 import register_vector  
from sentence_transformers import SentenceTransformer

# Configure logging to stdout 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
DB_HOST = os.getenv("DB_HOST", "vector_db")
DB_NAME = os.getenv("POSTGRES_DB", "ragdb")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

# Initialize Embedding Model
logger.info("Loading SentenceTransformer model from local image...")
model = SentenceTransformer('./model_data', device='cpu')
logger.info("Model loaded successfully.")

class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

class IngestRequest(BaseModel):
    documents: List[Document]

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        raise

@app.on_event("startup")
def startup_db():
    """Initialize DB schema on startup"""
    try:
        conn = get_db_connection()
        conn.autocommit = True
        
        # Register the vector type with psycopg2
        register_vector(conn)
        
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(384)
            )
        """)
        cur.close()
        conn.close()
        logger.info("✅ Database schema initialized.")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    logger.info(f"Received ingestion request for {len(request.documents)} documents.")
    
    conn = get_db_connection()
    # CRITICAL: Register vector type on this new connection
    register_vector(conn) 
    
    cur = conn.cursor()
    
    try:
        count = 0
        for doc in request.documents:
            logger.info(f"Processing doc: {doc.id}")
            
            # 1. Generate Embedding
            embedding = model.encode(doc.text).tolist()
            
            # 2. Serialize Metadata
            meta_json = json.dumps(doc.metadata)
            
            # 3. Upsert
            cur.execute("""
                INSERT INTO documents (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET content = EXCLUDED.content, 
                    metadata = EXCLUDED.metadata, 
                    embedding = EXCLUDED.embedding
            """, (doc.id, doc.text, meta_json, embedding))
            
            count += 1
            
        conn.commit()
        logger.info(f"Successfully indexed {count} documents.")
        return {"status": "success", "indexed": count}
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    # Ensure port matches Dockerfile and docker-compose
    uvicorn.run(app, host="0.0.0.0", port=8004)
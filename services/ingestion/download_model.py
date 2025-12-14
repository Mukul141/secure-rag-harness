import os
import shutil
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
OUTPUT_DIR = "./model_data"

# 1. Clean up any failed previous attempts
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

print(f"📉 Downloading {MODEL_NAME}...")

# 2. Download (letting it use default cache)
model = SentenceTransformer(MODEL_NAME, revision=REVISION)
model.save(OUTPUT_DIR)

print(f"✅ Model successfully baked into {OUTPUT_DIR}")
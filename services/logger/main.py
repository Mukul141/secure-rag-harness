from fastapi import FastAPI, Request
import json
import os

app = FastAPI()

# Ensure logs dir exists
os.makedirs("logs", exist_ok=True)

@app.post("/log")
async def log(request: Request):
    data = await request.json()
    with open("logs/results.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    return {"status": "logged"}
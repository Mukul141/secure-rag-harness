# ./services/*/Dockerfile
FROM python:3.11-slim-bookworm@sha256:917ec0e42cd6af87657a768449c2f604a6b67c7ab8e10ff917b8724799f816d3
WORKDIR /app

#Installing git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 1. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# 2. Bake the Model 
COPY download_model.py .
RUN python3 download_model.py

# 3. Copy Code
COPY . .

# 4. Run on Port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
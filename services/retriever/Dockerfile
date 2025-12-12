# ./services/*/Dockerfile
FROM python:3.11-slim-bookworm@sha256:917ec0e42cd6af87657a768449c2f604a6b67c7ab8e10ff917b8724799f816d3
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"] 
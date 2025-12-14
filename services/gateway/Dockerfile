# ./services/*/Dockerfile
FROM python:3.11-slim-bookworm@sha256:917ec0e42cd6af87657a768449c2f604a6b67c7ab8e10ff917b8724799f816d3

WORKDIR /app

#Install dependecies
COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy source code
COPY . .

#Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
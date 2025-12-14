FROM python:3.11-slim-bookworm
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model
COPY download_model.py .
RUN python3 download_model.py

# Copy app code
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
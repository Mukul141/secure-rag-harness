# Secure RAG Evaluation Harness

This repository contains the reproducibility harness for the paper **"Secure RAG Engineering"**. It allows engineering teams to benchmark defenses (EcoSafeRAG, ATM, Guardrails) against attacks (Poisoning, Prompt Injection) under strict SLA constraints.

## 🏗️ Architecture
The system follows a microservices architecture running in Docker:
1.  **Gateway:** API Entry point and Topology Router.
2.  **Retriever:** Hybrid search engine (BM25 + Dense).
3.  **Vector DB:** PostgreSQL with `pgvector` (Pinned Version).
4.  **Policy:** Middleware for input/output guardrails.
5.  **Logger:** Centralized telemetry sink for metrics.

## 🚀 Prerequisites

### Hardware
* **OS:** Ubuntu 22.04 (WSL2 on Windows or Native Linux).
* **GPU:** NVIDIA GPU with >= 6GB VRAM recommended (Tested on RTX 4060).
* **RAM:** 16GB System RAM minimum.

### Software Tools
* **Docker Desktop** (Enable WSL2 Integration).
* **Ollama** (Linux Version).
* **Make** (`sudo apt install make`).



## 🛠️ Quick Start

### 1. Setup Environment
Clone the repo and run the setup command. This checks for the necessary tools and creates your `.env` configuration file.

```bash
make setup
````

**⚠️ Important for WSL2 Users:**
The `.env` file will default to a standard host URL. If you are on WSL2, you **must** edit `.env` to point to your specific WSL IP address to allow Docker containers to reach Ollama.

1.  Find your IP: `ip addr show eth0`
2.  Edit `.env`:
    ```bash
    # Replace x.x.x.x with your eth0 IP
    LLM_API_BASE=[http://172.](http://172.)x.x.x.x:11434/v1
    ```

### 2\. Install & Serve the Model

We use a 4-bit quantized **Llama-3-8B** to fit in consumer memory.

```bash
# In Terminal A
ollama pull llama3
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

*Note: Keep Terminal A open.*

### 3\. Launch Infrastructure

Build and start the container stack.

```bash
# In Terminal B
make up
```

*First run will take a few minutes to download the pinned Docker images.*

### 4\. Run Smoke Test

Verify the pipeline is connected and generating answers.

```bash
make test
```

**Expected Output:**

```json
{"response": "Based on the provided context...", "context_used": [...], "model": "llama3"}
```

-----

## 🧪 Running Experiments

To run a full evaluation suite as described in the paper:

```bash
make run attack=prompt_injection defense=guardrails profile=P1
```

  * **Attacks:** `none`, `prompt_injection`, `retrieval_poisoning`, `opinion_manipulation`
  * **Defenses:** `baseline`, `guardrails`, `ecosafe`, `atm`, `skeptical`
  * **Profiles:** `P1` (SaaS), `P2` (VPC), `P3` (Regulated)

## 📦 Reproducibility Notes

  * **Determinism:** All services are pinned to specific SHA256 digests.
  * **OS Base:** All custom containers use `python:3.11-slim-bookworm` (Debian 12).
  * **Vector DB:** Uses `pgvector` (Postgres 16.2).

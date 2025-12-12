# Variables with defaults (can be overridden via CLI)
attack ?= none
defense ?= baseline
profile ?= P1
topology ?= sequential
seed ?= 42
limit ?= 10

.PHONY: setup up down test run help

# 1. Setup: Ensures environment configuration exists
setup:
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		echo "# Auto-generated .env" > .env; \
		echo "LLM_API_BASE=http://host.docker.internal:11434/v1" >> .env; \
		echo "LLM_MODEL_NAME=llama3" >> .env; \
		echo ""; \
		echo "⚠️  WARNING: If running on WSL2, you MUST update LLM_API_BASE in .env"; \
		echo "   to use your eth0 IP address (e.g., http://172.x.x.x:11434/v1)."; \
		echo "   Run 'ip addr show eth0' to find it."; \
	else \
		echo "✅ .env file already exists."; \
	fi

# 2. Infrastructure Management
up: setup
	docker-compose up --build -d

down:
	docker-compose down

# 3. Smoke Test (Manual verification)
test:
	curl -X POST http://localhost:8000/chat \
	-H "Content-Type: application/json" \
	-d '{"query": "Hello RAG", "topology": "sequential"}'

# 4. The Experiment Runner (The "Harness")
# This maps the Paper's CLI syntax to your Python script
run:
	python3 harness/main.py \
		--attack=$(attack) \
		--defense=$(defense) \
		--profile=$(profile) \
		--topology=$(topology) \
		--seed=$(seed) \
		--limit=$(limit)

# Help command to show usage
help:
	@echo "Secure RAG Harness - Usage:"
	@echo "  make setup       Create config files"
	@echo "  make up          Start infrastructure (Gateway, DB, Retriever)"
	@echo "  make test        Run a single manual smoke test"
	@echo "  make run         Execute an experiment suite"
	@echo ""
	@echo "Experiment Arguments (defaults shown):"
	@echo "  attack=none      [none, prompt_injection, poisoning, ...]"
	@echo "  defense=baseline [baseline, guardrails, ecosafe, atm ...]"
	@echo "  profile=P1       [P1, P2, P3]"
	@echo "  topology=seq     [sequential, branching, loop]"
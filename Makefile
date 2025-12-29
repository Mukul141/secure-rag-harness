# Default variables (can be overridden from the command line)
attack ?= none
defense ?= baseline
profile ?= P1
topology ?= sequential
seed ?= 42
limit ?= 10

.PHONY: setup setup-llm setup-data up down test run help

# --------------------------------------------------
# Setup targets
# --------------------------------------------------

# Ensures the environment configuration and LLM setup exist
setup: setup-llm
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		echo "# Auto-generated .env" > .env; \
		echo "LLM_API_BASE=http://host.docker.internal:11434/v1" >> .env; \
		echo "LLM_MODEL_NAME=secure-rag-llama3" >> .env; \
		echo ""; \
		echo "WARNING: If running on WSL2, update LLM_API_BASE in .env"; \
		echo "to use your eth0 IP (run 'ip addr show eth0')."; \
	else \
		echo ".env file already exists."; \
	fi

# Builds the local Ollama model if it does not already exist
setup-llm:
	@# Check if the Mistral model is already registered in Ollama
	@if ollama list | grep -q "mistral-7b-instruct:q4km"; then \
		echo "Model 'mistral-7b-instruct:q4km' already exists. Skipping build."; \
	else \
		echo "Configuring local Mistral LLM..."; \
		\
		# Verify Mistral GGUF weights exist \
		if [ ! -f services/llm/weights/mistral-7b-instruct-v0.2.Q4_K_M.gguf ]; then \
			echo "GGUF weights not found at services/llm/weights/mistral-7b-instruct-v0.2.Q4_K_M.gguf"; \
			echo "Please run the download command from the README."; \
			exit 1; \
		fi; \
		\
		# Generate Modelfile with absolute Linux paths (WSL-safe) \
		echo "Generating Modelfile with absolute paths..."; \
		sed "s|__WEIGHTS_DIR__|$(PWD)/services/llm/weights|g" \
			services/llm/Modelfile.template > services/llm/Modelfile; \
		\
		# Create the Ollama model \
		echo "Creating Mistral model: "; \
		ollama create mistral-7b-instruct:q4km -f services/llm/Modelfile; \
		echo "Model 'mistral-7b-instruct:q4km' created successfully."; \
	fi


# Installs data dependencies and downloads benchmark datasets
setup-data:
	@echo "Installing data dependencies..."
	pip install -r data/requirements.txt
	@echo "Running unified download script..."
	python3 data/scripts/download_datasets.py

# --------------------------------------------------
# Infrastructure management
# --------------------------------------------------

up: setup
	docker-compose up --build -d

down:
	docker-compose down

# --------------------------------------------------
# Testing and execution
# --------------------------------------------------

# Runs a basic manual smoke test against the API
test:
	curl -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"query": "Hello RAG", "topology": "sequential"}'

# Executes the main experiment harness
run:
	python3 harness/main.py \
		--attack=$(attack) \
		--defense=$(defense) \
		--profile=$(profile) \
		--topology=$(topology) \
		--seed=$(seed) \
		--limit=$(limit)

# Convenience targets for prompt-injection experiments
run-pi-direct:
	python3 harness.main \
		--attack=pi-direct \
		--payload_type=combined \
		--limit=10

run-pi-indirect:
	python3 harness.main \
		--attack=pi-indirect \
		--payload_type=combined \
		--limit=10

# Displays available make commands
help:
	@echo "Secure RAG Harness - Usage:"
	@echo "  make setup       Create config files and build the LLM"
	@echo "  make up          Start infrastructure"
	@echo "  make test        Run a manual smoke test"
	@echo "  make run         Execute an experiment suite"

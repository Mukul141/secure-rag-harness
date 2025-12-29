import time
import requests
import json
import pandas as pd
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

from .base_experiment import BaseExperiment
from harness.evaluator  import PIEvaluator


class PromptInjectionExperiment(BaseExperiment, ABC):
    def run(self):
        """
        Entry point for prompt injection experiments.
        """
        print(f"Running prompt injection experiment: {self.config['attack_type']}")

        # Initialize the evaluator (expects a local Ollama instance)
        self.evaluator = PIEvaluator(llm_host="http://localhost:11434/v1")

        dataset = self._load_dataset()

        # Build the attack queue (payloads and documents are prepared here)
        attack_queue = self._build_attack_queue(dataset)

        # Execute the experiment loop
        self._execute_loop(attack_queue)

    def _load_dataset(self):
        data_path = "data/corpus/natural_questions_sample.json"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Dataset not found. Run 'make setup-data' first."
            )

        with open(data_path, "r") as f:
            return json.load(f)[: self.config["limit"]]

    @abstractmethod
    def _build_attack_queue(self, dataset):
        """
        Must be implemented by Direct and Indirect variants.
        """
        pass

    def _execute_loop(self, attack_queue):
        print(f"Starting attack loop for {len(attack_queue)} samples...")
        results = []

        # Perform a single bulk ingestion of all documents used in the attack
        # This replaces per-sample isolation and improves throughput
        all_docs = [packet[4] for packet in attack_queue if packet[4] is not None]
        if all_docs:
            print(f"Bulk ingesting {len(all_docs)} documents...")
            self.reset_and_ingest(all_docs)

        for i, packet in enumerate(tqdm(attack_queue, desc="Attacking")):
            original_item, payload_name, prompt_text, search_text, _ = packet
            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.gateway_host}/chat",
                    json={
                        "query": prompt_text,
                        "search_query": search_text,
                        "k": 1,
                        "topology": self.config["topology"],
                        "profile": self.config["profile"],
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()

                response_text = data["response"]
                retrieved_context = data.get("context", [])

                # Extract retrieval metadata if a document was returned
                hit_id = retrieved_context[0]["id"] if retrieved_context else None
                source_scores = (
                    retrieved_context[0].get("source_scores", {})
                    if retrieved_context
                    else {}
                )

                # Verify whether the expected document was retrieved
                is_correct_hit = (
                    hit_id is not None and original_item["id"] in hit_id
                )

                # Evaluate the model response using the judge
                evaluation = self.evaluator.evaluate(response_text)

                results.append(
                    {
                        "type": self.config["attack_type"],
                        "strategy": payload_name,
                        "query": original_item["query"],
                        "asv": evaluation["score"],
                        "reason": evaluation["reasoning"],
                        "retrieved_id": hit_id,
                        "retrieval_correct": is_correct_hit,
                        "dense_rank": source_scores.get("dense_rank"),
                        "sparse_rank": source_scores.get("sparse_rank"),
                        "latency": time.time() - start_time,
                    }
                )

            except Exception as exc:
                print(f"Error on sample {i}: {exc}")

        self._save_results(results)

    def _save_results(self, results):
        df = pd.DataFrame(results)
        os.makedirs(self.config["output_dir"], exist_ok=True)

        csv_path = (
            f"{self.config['output_dir']}/results_{self.config['attack_type']}.csv"
        )
        df.to_csv(csv_path, index=False)

        print(f"Results saved to: {csv_path}")

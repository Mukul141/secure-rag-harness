from abc import ABC, abstractmethod
import requests
import time


class BaseExperiment(ABC):
    def __init__(self, config):
        self.config = config
        self.ingest_host = "http://localhost:8004"
        self.gateway_host = "http://localhost:8000"

    @abstractmethod
    def run(self):
        """
        Executes the full experiment lifecycle.
        """
        pass

    def reset_and_ingest(self, documents):
        """
        Resets the vector database and ingests a new set of documents.
        """

        # Reset the database
        reset_url = f"{self.ingest_host}/reset"
        print(f"Resetting vector database at {reset_url}...")

        try:
            response = requests.post(reset_url, timeout=10)
            response.raise_for_status()
            print("Database reset completed.")
        except Exception as exc:
            print(f"Critical error: failed to reset database. Error: {exc}")
            raise

        # Small delay to allow DB locks to clear
        time.sleep(0.1)

        # Ingest documents in batches
        ingest_url = f"{self.ingest_host}/ingest"
        print(f"Ingesting {len(documents)} documents to {ingest_url}...")

        batch_size = 50
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]

            payload = {
                "documents": [
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                    }
                    for doc in batch
                ]
            }

            try:
                response = requests.post(
                    ingest_url,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
            except Exception as exc:
                print(
                    f"Ingestion failed for batch starting at index {start}. "
                    f"Error: {exc}"
                )
                raise

        print("Ingestion completed successfully.")


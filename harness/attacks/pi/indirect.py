from .payloads import get_generator, get_all_generators
from .base import PromptInjectionExperiment


class IndirectPromptInjectionExperiment(PromptInjectionExperiment):
    def _build_attack_queue(self, dataset):
        print("Preparing poisoned documents for indirect prompt injection...")
        poisoned_docs = []
        attack_queue = []
        generators = get_all_generators()

        for i, item in enumerate(dataset):
            if self.config["payload_type"] == "mixed":
                payload_name, payload_gen = generators[i % len(generators)]
            else:
                payload_name = self.config["payload_type"]
                payload_gen = get_generator(payload_name)

            # Create a poisoned copy of the original document
            poisoned_item = item.copy()
            poisoned_item["id"] = f"{item['id']}_{payload_name}"
            poisoned_item["text"] = payload_gen.inject(item["text"])

            poisoned_docs.append(poisoned_item)

            # Packet format:
            # (original_item, payload_name, prompt_text, search_query)
            attack_queue.append(
                (item, payload_name, item["query"], item["query"])
            )

        # Reset the database once and ingest all poisoned documents
        self.reset_and_ingest(poisoned_docs)

        return attack_queue

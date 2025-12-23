from .payloads import get_generator, get_all_generators
from .base import PromptInjectionExperiment


class DirectPromptInjectionExperiment(PromptInjectionExperiment):
    def _build_attack_queue(self, dataset):
        print("Preparing attack queue with clean documents (Isolation Mode)...")
        attack_queue = []
        generators = get_all_generators()

        for i, item in enumerate(dataset):
            if self.config["payload_type"] == "mixed":
                payload_name, payload_gen = generators[i % len(generators)]
            else:
                payload_name = self.config["payload_type"]
                payload_gen = get_generator(payload_name)

            # Inject the payload into the user query
            poisoned_query = payload_gen.inject(item["query"])

            # Packet format:
            # (original_item, payload_name, prompt_text, search_query, target_doc)
            # For direct PI, the doc is clean, but we still isolate it
            attack_queue.append(
                (item, payload_name, poisoned_query, item["query"], item)
            )

        return attack_queue
from .payloads import get_generator, get_all_generators
from .base import PromptInjectionExperiment


class IndirectPromptInjectionExperiment(PromptInjectionExperiment):
    def _build_attack_queue(self, dataset):
        print("Preparing attack queue with poisoned documents (Isolation Mode)...")
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

            # Packet format:
            # (original_item, payload_name, prompt_text, search_query, target_doc)
            attack_queue.append(
                (item, payload_name, item["query"], item["query"], poisoned_item)
            )

        return attack_queue
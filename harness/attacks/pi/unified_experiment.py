import os
import random
import pandas as pd
import requests
from abc import ABC
from tqdm import tqdm

from harness.attacks.pi.base import BaseExperiment
from harness.attacks.pi.payloads import get_all_generators
from harness.evaluator.PIEvaluator import PIEvaluator
from harness.tasks.config import TASK_CONFIGS
from harness.tasks.loader import load_dataset


class UnifiedPIExperiment(BaseExperiment, ABC):
    """
    Executes the full privilege-separated prompt injection benchmark.
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_cache = {}
        self.results = []
        self.evaluator = PIEvaluator()
        self.payload_generators = get_all_generators()

    def run(self):
        """
        Loads all datasets and evaluates all target→injected pairs.
        """
        print(f"Starting Unified PI Benchmark ({self.config['limit']} samples per pair)")

        for task_name, cfg in tqdm(TASK_CONFIGS.items(), desc="Loading datasets"):
            self.data_cache[task_name] = load_dataset(task_name, cfg)

        task_names = list(TASK_CONFIGS.keys())
        total_pairs = len(task_names) ** 2

        with tqdm(total=total_pairs, desc="Task matrix") as pbar:
            for target_task in task_names:
                for injected_task in task_names:

                    # Self-injection omitted in this benchmark configuration
                    if target_task == injected_task:
                        pbar.update(1)
                        continue

                    pbar.set_description(f"{target_task} -> {injected_task}")
                    self._run_pair(target_task, injected_task)
                    pbar.update(1)

        self._save_results()

    def _run_pair(self, target_task, injected_task):
        """
        Executes all payload-variant attacks for a specific pair.
        """
        target_samples, _ = self._select_samples(target_task, self.config["limit"])
        injected_samples, _ = self._select_samples(injected_task, self.config["limit"])

        if not target_samples or not injected_samples:
            return

        for idx, (target_item, injected_item) in enumerate(zip(target_samples, injected_samples)):
            name, generator = self.payload_generators[idx % len(self.payload_generators)]

            injected_instruction = TASK_CONFIGS[injected_task]["injected_instruction"]
            injected_text = injected_item["text"]
            full_injection = f"{injected_instruction} {injected_text}"

            poisoned_context = generator.inject(full_injection)

            self._execute_chat(
                target_task=target_task,
                injected_task=injected_task,
                target_item=target_item,
                injected_item=injected_item,
                poisoned_context=poisoned_context,
                attack_method=name,
            )

    def _execute_chat(self, target_task, injected_task, target_item, injected_item, poisoned_context, attack_method):
        """
        Performs the two-step evaluation:
        1. Golden run for injected-task capability.
        2. Attack run with target task as system and poisoned context as user.
        """
        try:
            golden_response = None

            # Golden run (only needed for generative injected tasks)
            try:
                golden_req = {
                    "system_prompt": TASK_CONFIGS[injected_task]["instruction"],
                    "query": f"Context: {injected_item['text']}",
                    "topology": "pi",
                    "profile": self.config["profile"],
                }

                golden_res = requests.post(
                    f"{self.gateway_host}/chat",
                    json=golden_req,
                    timeout=120,
                )

                if golden_res.status_code == 200:
                    golden_response = golden_res.json().get("response", "")

            except Exception:
                pass

            attack_req = {
                "system_prompt": TASK_CONFIGS[target_task]["instruction"],
                "query": f"Context: {poisoned_context}",
                "topology": "pi",
                "profile": self.config["profile"],
            }

            attack_res = requests.post(
                f"{self.gateway_host}/chat",
                json=attack_req,
                timeout=120,
            )
            attack_res.raise_for_status()

            attack_output = attack_res.json().get("response", "")

            # Label normalization
            raw_label = injected_item.get("label")
            label_map = TASK_CONFIGS[injected_task].get("label_map")

            if label_map is not None and raw_label in label_map:
                injected_label = label_map[raw_label]
            else:
                injected_label = str(raw_label) if raw_label is not None else None

            scores = self.evaluator.evaluate(
                target_task=target_task,
                injected_task=injected_task,
                response=attack_output,
                injected_label=injected_label,
                golden_response=golden_response,
            )

            self.results.append({
                "target_task": target_task,
                "injected_task": injected_task,
                "attack_method": attack_method,
                "asv": scores["asv"],
                "asr": scores["asr"],
                "metric": scores["metric"],
                "response_length": len(attack_output),
                "golden_response_snippet": (golden_response or "")[:50],
                "response_snippet": attack_output[:50],
            })

        except Exception as exc:
            print(f"Error executing {target_task}-{injected_task}: {exc}")

    def _select_samples(self, task_name, count):
        """
        Random sampling for each task’s dataset.
        """
        dataset = self.data_cache[task_name]

        if len(dataset) <= count:
            return dataset, list(range(len(dataset)))

        indices = random.sample(range(len(dataset)), count)
        return [dataset[i] for i in indices], indices

    def _save_results(self):
        """
        Writes all evaluations to disk.
        """
        os.makedirs(self.config["output_dir"], exist_ok=True)

        path = os.path.join(self.config["output_dir"], "unified_pi_results.csv")
        pd.DataFrame(self.results).to_csv(path, index=False)

        print(f"Saved results to {path}")

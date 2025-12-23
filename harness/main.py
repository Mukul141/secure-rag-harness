import fire

from attacks.pi.direct import DirectPromptInjectionExperiment
from attacks.pi.indirect import IndirectPromptInjectionExperiment

# Registry of supported experiments
EXPERIMENTS = {
    "pi-direct": DirectPromptInjectionExperiment,
    "pi-indirect": IndirectPromptInjectionExperiment,
}


def main(
    attack="pi-direct",       # Attack family
    payload_type="combined",  # Attack variant (PI-specific)
    topology="sequential",
    profile="P1",
    seed=42,
    limit=10,
    output_dir="results",
):
    # Validate selected attack
    if attack not in EXPERIMENTS:
        print(f"Unknown attack: {attack}. Available options: {list(EXPERIMENTS.keys())}")
        return

    # Build experiment configuration
    config = {
        "attack_type": attack,
        "payload_type": payload_type,
        "topology": topology,
        "profile": profile,
        "seed": seed,
        "limit": limit,
        "output_dir": output_dir,
    }

    # Instantiate and run the selected experiment
    experiment_cls = EXPERIMENTS[attack]
    experiment = experiment_cls(config)
    experiment.run()


if __name__ == "__main__":
    fire.Fire(main)

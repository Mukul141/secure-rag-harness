import json
from harness.attacks.pi.payloads import get_all_generators


def inspect_poisoned_docs():
    # Load a small sample from the clean corpus
    data_path = "data/corpus/natural_questions_sample.json"
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Use the first document for inspection
    sample = dataset[0]

    print("ORIGINAL TEXT (First 100 chars):")
    print(f"{sample['text'][:100]}...\n")
    print("-" * 60)

    # Apply each payload generator and inspect the output
    generators = get_all_generators()

    for name, generator in generators:
        poisoned_text = generator.inject(sample["text"])

        print(f"\nSTRATEGY: {name.upper()}")

        # Most payloads appear near the end of the document.
        # xml_cloak is an exception, so we preview the beginning instead.
        if name == "xml_cloak":
            print("PREVIEW (Start):")
            print(poisoned_text[:300])
        else:
            print("PREVIEW (End):")
            print(f"...{poisoned_text[-300:]}")

    print("\n" + "-" * 60)
    print(
        "If the 'PWNED_SECURE_RAG' trigger appears above, "
        "the injection logic is functioning correctly."
    )


if __name__ == "__main__":
    inspect_poisoned_docs()

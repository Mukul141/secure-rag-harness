import os
import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_FILE = "data/corpus/natural_questions_sample.json"
LIMIT = 1000


def download_data():
    print("Downloading unique SQuAD v2.0 contexts...")

    # Load the SQuAD v2 training split
    dataset = load_dataset("squad_v2", split="train", streaming=False)

    samples = []
    seen_contexts = set()  # Tracks unique context paragraphs

    print("Processing documents...")
    for row in tqdm(dataset):
        if len(samples) >= LIMIT:
            break

        context = row["context"]

        # Skip duplicate paragraphs
        if context in seen_contexts:
            continue

        answers = row["answers"]["text"]

        # Skip entries without answers (ground truth is required)
        if not answers:
            continue

        seen_contexts.add(context)

        samples.append(
            {
                "id": f"doc_{len(samples)}",
                "query": row["question"],
                "text": context,
                "answer": answers[0],
                "source": "squad_v2",
                "metadata": {
                    "title": row["title"],
                },
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} unique paragraphs to {OUTPUT_FILE}")

    if len(samples) > 1:
        print(f"Sample 0 title: {samples[0]['metadata']['title']}")
        print(f"Sample 1 title: {samples[1]['metadata']['title']}")


if __name__ == "__main__":
    download_data()

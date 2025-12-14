import json
import argparse
import random
from pathlib import Path

def generate_corpus(num_docs, seed, output_file):
    random.seed(seed)
    data = []
    
    topics = ["Cybersecurity", "Artificial Intelligence", "Quantum Computing", "Biology", "History"]
    
    for i in range(num_docs):
        topic = random.choice(topics)
        doc_id = f"doc_{i}"
        # A simple template to create "retrievable" text
        text = f"This is a document about {topic}. The specific fact ID is {random.randint(1000, 9999)}. Secure RAG testing data."
        
        data.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "source": "synthetic",
                "topic": topic,
                "tenant_id": "tenant_a" if i % 2 == 0 else "tenant_b" # For P1 isolation tests
            }
        })

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {"documents": data}

    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    
    print(f"✅ Generated {num_docs} documents to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/corpus/synthetic.json")
    args = parser.parse_args()
    
    generate_corpus(args.count, args.seed, args.output)
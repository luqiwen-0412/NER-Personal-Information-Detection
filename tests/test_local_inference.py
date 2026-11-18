# Local CPU Inference Test
from pathlib import Path
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

# Identify project and model directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "final_model"

# Configure CPU execution
device = torch.device("cpu")

print("Local BERT NER Inference Test")
print("------------------------------------")
print("Model directory:", MODEL_DIR)
print("Execution device:", device)

# Load the saved tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

# Load our fine-tuned model
model = AutoModelForTokenClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model = model.to(device)
model.eval()

# Create the NER inference pipeline
ner_pipeline = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first",
    device=-1
)

# Prepare test sentences
test_sentences = [
    "John Smith works at Microsoft in London.",

    "Ahmad bin Abdullah lives in Johor Bahru, Malaysia.",

    "Siti Nurhaliza studied at Universiti Teknologi Malaysia.",

    "Barack Obama visited Google headquarters in California.",

    "The weather is pleasant and the flowers are beautiful."
]

# Required AIROST entity categories
required_categories = {"PER", "LOC", "ORG"}

# Perform inference
for index, sentence in enumerate(
    test_sentences,
    start=1
):

    print("\n" + "=" * 60)
    print(f"Test Case {index}")
    print("=" * 60)

    print("Input:", sentence)

    start_time = time.perf_counter()

    with torch.inference_mode():
        predictions = ner_pipeline(sentence)

    inference_time = (
        time.perf_counter() - start_time
    )

    # Display all detected entities
    print("\nAll Detected Entities:")

    if not predictions:
        print("No named entities detected.")

    for entity in predictions:

        entity_text = sentence[
            entity["start"]:entity["end"]
        ]

        print(
            f"Entity: {entity_text} | "
            f"Category: {entity['entity_group']} | "
            f"Confidence: {entity['score'] * 100:.2f}%"
        )

    # Filter AIROST required categories
    filtered_entities = [
        entity
        for entity in predictions
        if entity["entity_group"] in required_categories
    ]

    print("\nAIROST Required Entities:")

    if not filtered_entities:
        print("No required entities detected.")

    for entity in filtered_entities:

        entity_text = sentence[
            entity["start"]:entity["end"]
        ]

        print(
            f"{entity_text} -> "
            f"{entity['entity_group']}"
        )

    print(
        f"\nInference time: "
        f"{inference_time:.4f} seconds"
    )

print("\n" + "=" * 60)
print("All local inference tests completed!")
print("=" * 60)
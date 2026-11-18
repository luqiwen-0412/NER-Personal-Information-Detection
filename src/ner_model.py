"""
Reusable NER model and inference module.

Loads the project's fine-tuned BERT model and performs local CPU-based named entity recognition.
"""

from pathlib import Path
from functools import lru_cache
from pprint import pprint

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)


# Identify the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Define the fine-tuned model directory
MODEL_DIR = PROJECT_ROOT / "models" / "final_model"


# Configure CPU execution
DEVICE = torch.device("cpu")


# Define the required model files
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json"
)


# Define the required AIROST entity categories
REQUIRED_CATEGORIES = {"PER", "LOC", "ORG"}


def verify_model_files():
    """
    Verify that the required local model files exist.
    """

    for filename in REQUIRED_FILES:
        file_path = MODEL_DIR / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required model file missing: {file_path}"
            )


@lru_cache(maxsize=1)
def load_ner_model():
    """
    Load and cache the fine-tuned BERT NER pipeline.

    Returns:
        A Hugging Face token classification pipeline.
    """

    # Verify the required model files
    verify_model_files()

    # Load the saved tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    # Load our fine-tuned BERT model
    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    # Move the model to the CPU
    model = model.to(DEVICE)

    # Configure the model for inference
    model.eval()

    # Create the NER inference pipeline
    ner_pipeline = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=-1
    )

    return ner_pipeline


def detect_entities(text: str) -> list[dict]:
    """
    Detect person, location and organisation entities.

    Args:
        text: Original raw input text.

    Returns:
        A list of dictionaries containing entity text,
        label, confidence and character offsets.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the text exceeds model capacity.
    """

    # Validate the input data type
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Handle empty or whitespace-only input
    if not text.strip():
        return []

    # Load or reuse the cached NER pipeline
    ner_pipeline = load_ner_model()

    # Determine the model's maximum input length
    max_tokens = min(
        ner_pipeline.tokenizer.model_max_length,
        ner_pipeline.model.config.max_position_embeddings
    )

    # Count input tokens without truncating the text
    token_ids = ner_pipeline.tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    token_count = len(token_ids)

    # Prevent silent truncation of long input
    if token_count > max_tokens:
        raise ValueError(
            f"Input contains {token_count} tokens, "
            f"but the model supports a maximum of "
            f"{max_tokens} tokens. "
            "Please enter a shorter text."
        )

    # Perform NER inference
    with torch.inference_mode():
        predictions = ner_pipeline(text)

    # Create an empty list for structured results
    detected_entities = []

    # Process the model predictions
    for entity in predictions:

        label = entity["entity_group"]

        # Exclude categories not required by AIROST
        if label not in REQUIRED_CATEGORIES:
            continue

        # Obtain character offsets
        start = int(entity["start"])
        end = int(entity["end"])

        # Verify that the character offsets are valid
        if not 0 <= start < end <= len(text):
            raise RuntimeError(
                "The model returned invalid entity offsets."
            )

        # Extract the entity from the original text
        entity_text = text[start:end]

        # Convert the confidence score to a Python float
        confidence = float(entity["score"])

        # Create a structured entity dictionary
        entity_result = {
            "text": entity_text,
            "label": label,
            "confidence": confidence,
            "start": start,
            "end": end
        }

        # Add the entity to the result list
        detected_entities.append(entity_result)

    return detected_entities


# Standalone module verification
if __name__ == "__main__":

    print("Reusable NER Model and Inference")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)
    print("Model directory:", MODEL_DIR)

    # Load the NER pipeline
    ner_pipeline = load_ner_model()

    print("Model architecture:",
          type(ner_pipeline.model).__name__)

    print("Model device:",
          next(ner_pipeline.model.parameters()).device)

    print("Number of labels:",
          ner_pipeline.model.config.num_labels)

    print("Evaluation mode:",
          not ner_pipeline.model.training)

    # Verify cached pipeline reuse
    cached_pipeline = load_ner_model()

    print("Cache reuse:",
          ner_pipeline is cached_pipeline)

    # Prepare a sample input
    sample_text = (
        "John Smith works at Microsoft in London."
    )

    print("\nInput text:")
    print(sample_text)

    # Perform entity detection
    results = detect_entities(sample_text)

    print("\nStructured NER Results")
    print("-" * 60)

    pprint(results, sort_dicts=False)

    print("\nTotal detected entities:", len(results))

    # Verify the expected entities and offsets
    expected_entities = {
        ("John Smith", "PER", 0, 10),
        ("Microsoft", "ORG", 20, 29),
        ("London", "LOC", 33, 39)
    }

    actual_entities = {
        (
            entity["text"],
            entity["label"],
            entity["start"],
            entity["end"]
        )
        for entity in results
    }

    assert actual_entities == expected_entities, (
        "Entity detection verification failed."
    )

    assert len(results) == 3, (
        "Unexpected number of entities."
    )

    print("\nEntity detection verification: PASSED")

    # Verify empty input handling
    assert detect_entities("") == []
    assert detect_entities("   ") == []

    print("Empty input verification: PASSED")

    # Verify invalid input handling
    try:
        detect_entities(123)

    except TypeError:
        print("Invalid input verification: PASSED")

    else:
        raise AssertionError(
            "Invalid input verification failed."
        )

    # Verify long input protection
    try:
        detect_entities("hello " * 600)

    except ValueError:
        print("Long input protection: PASSED")

    else:
        raise AssertionError(
            "Long input protection failed."
        )

    print("\nCache information:")
    print(load_ner_model.cache_info())

    print("\nAll raw text inference checks PASSED!")
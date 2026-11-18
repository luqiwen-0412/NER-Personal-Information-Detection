"""
Integration test for the reusable NER model module.
Verifies module importing, model caching, entity
prediction and character offsets.
"""

import time

import torch

from src.ner_model import load_ner_model


def main():

    print("Reusable NER Model Integration Test")
    print("=" * 60)

    # Load the reusable NER pipeline
    start_time = time.perf_counter()

    ner_pipeline = load_ner_model()

    loading_time = time.perf_counter() - start_time

    print(f"Model loading time: {loading_time:.2f} seconds")

    # Verify cached pipeline reuse
    cached_pipeline = load_ner_model()

    assert ner_pipeline is cached_pipeline, (
        "Model caching verification failed."
    )

    print("Model cache verification: PASSED")

    # Prepare a sample sentence
    text = "John Smith works at Microsoft in London."

    print("\nInput text:")
    print(text)

    # Perform NER inference
    start_time = time.perf_counter()

    with torch.inference_mode():
        predictions = ner_pipeline(text)

    inference_time = time.perf_counter() - start_time

    print("\nDetected Entities")
    print("-" * 60)

    detected_entities = set()

    # Process each detected entity
    for entity in predictions:

        start = entity["start"]
        end = entity["end"]

        # Verify valid character offsets
        assert 0 <= start < end <= len(text), (
            "Invalid entity character offsets."
        )

        # Extract the original entity text
        entity_text = text[start:end]

        label = entity["entity_group"]
        confidence = float(entity["score"])

        detected_entities.add((entity_text, label))

        print(f"Entity: {entity_text}")
        print(f"Category: {label}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Start index: {start}")
        print(f"End index: {end}")
        print("-" * 60)

    # Define expected entities
    expected_entities = {
        ("John Smith", "PER"),
        ("Microsoft", "ORG"),
        ("London", "LOC")
    }

    # Compare actual and expected predictions
    assert detected_entities == expected_entities, (
        f"Unexpected predictions: {detected_entities}"
    )

    print("\nEntity prediction verification: PASSED")
    print("Character offset verification: PASSED")

    print(f"Inference time: {inference_time:.4f} seconds")

    print("\nCache information:")
    print(load_ner_model.cache_info())

    print("\nAll reusable NER model tests PASSED!")


if __name__ == "__main__":
    main()
"""
Reusable automatic text redaction module.

Replaces detected person, location and organisation
entities with their corresponding category labels.
"""


# Define the replacement labels
REDACTION_LABELS = {
    "PER": "[PERSON]",
    "LOC": "[LOCATION]",
    "ORG": "[ORGANISATION]"
}


def redact_text(text: str, entities: list[dict]) -> str:
    """
    Redact detected named entities from the original text.

    Args:
        text: Original raw input text.
        entities: Structured entities returned by
                  detect_entities().

    Returns:
        The redacted text.

    Raises:
        TypeError: If the input types are incorrect.
        ValueError: If entity information is invalid
                    or contains overlapping offsets.
    """

    # Validate the input text
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Validate the entity list
    if not isinstance(entities, list):
        raise TypeError("Entities must be a list.")

    # Return the original text if no entities exist
    if not entities:
        return text

    # Validate every entity before redaction
    for entity in entities:

        if not isinstance(entity, dict):
            raise TypeError(
                "Each entity must be a dictionary."
            )

        required_keys = {"text", "label", "start", "end"}

        if not required_keys.issubset(entity):
            raise ValueError(
                "Entity information is incomplete."
            )

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        # Validate offset data types
        if type(start) is not int or type(end) is not int:
            raise ValueError(
                "Entity offsets must be integers."
            )

        # Validate character offset boundaries
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"Invalid entity offsets: {start}, {end}"
            )

        # Validate the entity category
        if not isinstance(label, str) or label not in REDACTION_LABELS:
            raise ValueError(
                f"Unsupported entity label: {label}"
            )

        # Verify entity text against the original input
        if entity["text"] != text[start:end]:
            raise ValueError(
                "Entity text does not match its offsets."
            )

    # Sort entities from right to left
    sorted_entities = sorted(
        entities,
        key=lambda entity: entity["start"],
        reverse=True
    )

    # Check for overlapping entities
    previous_start = len(text)

    for entity in sorted_entities:

        if entity["end"] > previous_start:
            raise ValueError(
                "Overlapping entity offsets detected."
            )

        previous_start = entity["start"]

    # Preserve the original input
    redacted_text = text

    # Replace each entity from right to left
    for entity in sorted_entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        replacement = REDACTION_LABELS[label]

        redacted_text = (
            redacted_text[:start]
            + replacement
            + redacted_text[end:]
        )

    return redacted_text


# Standalone redaction verification
if __name__ == "__main__":

    print("Automatic Text Redaction Verification")
    print("=" * 60)

    # Prepare sample text
    sample_text = (
        "John Smith works at Microsoft in London."
    )

    # Prepare mock entity predictions
    sample_entities = [
        {
            "text": "John Smith",
            "label": "PER",
            "confidence": 0.9986,
            "start": 0,
            "end": 10
        },
        {
            "text": "Microsoft",
            "label": "ORG",
            "confidence": 0.9987,
            "start": 20,
            "end": 29
        },
        {
            "text": "London",
            "label": "LOC",
            "confidence": 0.9989,
            "start": 33,
            "end": 39
        }
    ]

    # Perform automatic redaction
    result = redact_text(sample_text, sample_entities)

    print("\nOriginal text:")
    print(sample_text)

    print("\nRedacted text:")
    print(result)

    # Define the expected result
    expected_text = (
        "[PERSON] works at [ORGANISATION] "
        "in [LOCATION]."
    )

    # Verify automatic redaction
    assert result == expected_text, (
        "Automatic redaction verification failed."
    )

    print("\nAutomatic redaction: PASSED")

    # Verify empty entity handling
    assert redact_text(sample_text, []) == sample_text

    print("Empty entity list handling: PASSED")

    # Verify empty text handling
    assert redact_text("", []) == ""

    print("Empty text handling: PASSED")

    # Verify unsorted entity handling
    unsorted_entities = [
        sample_entities[2],
        sample_entities[0],
        sample_entities[1]
    ]

    assert redact_text(
        sample_text,
        unsorted_entities
    ) == expected_text

    print("Unsorted entity handling: PASSED")

    print("\nAll basic redaction checks PASSED!")
"""
Integration testing for NER and automatic text redaction.

Uses actual predictions from the project's fine-tuned
BERT model instead of mock entity dictionaries.
"""

import time

from src.ner_model import (
    load_ner_model,
    detect_entities
)

from src.redaction import (
    redact_text,
    REDACTION_LABELS
)


def build_expected_redaction(text, entities):
    """
    Independently construct the expected redacted text
    using left-to-right processing.
    """

    parts = []
    cursor = 0

    sorted_entities = sorted(
        entities,
        key=lambda entity: entity["start"]
    )

    for entity in sorted_entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        # Preserve text before the detected entity
        parts.append(text[cursor:start])

        # Insert the replacement label
        parts.append(REDACTION_LABELS[label])

        # Move the cursor beyond the current entity
        cursor = end

    # Preserve all remaining text
    parts.append(text[cursor:])

    return "".join(parts)


def main():

    print("NER and Redaction Integration Test")
    print("=" * 70)

    # Load the trained model before testing
    start_time = time.perf_counter()

    load_ner_model()

    loading_time = time.perf_counter() - start_time

    print(f"Model loading time: {loading_time:.4f} seconds")

    # Define the integration test cases
    test_cases = [
        {
            "name": "All three entity categories",
            "text": (
                "John Smith works at Microsoft in London."
            ),
            "expected": (
                "[PERSON] works at [ORGANISATION] "
                "in [LOCATION]."
            )
        },
        {
            "name": "Malaysian person and locations",
            "text": (
                "Ahmad bin Abdullah lives in "
                "Johor Bahru, Malaysia."
            ),
            "expected": None
        },
        {
            "name": "Malaysian person and organisation",
            "text": (
                "Siti Nurhaliza studied at "
                "Universiti Teknologi Malaysia."
            ),
            "expected": None
        },
        {
            "name": "International named entities",
            "text": (
                "Barack Obama visited Google "
                "headquarters in California."
            ),
            "expected": None
        },
        {
            "name": "Repeated person names",
            "text": (
                "John Smith met John Smith in London."
            ),
            "expected": None
        },
        {
            "name": "No named entities",
            "text": (
                "The weather is pleasant "
                "and the flowers are beautiful."
            ),
            "expected": (
                "The weather is pleasant "
                "and the flowers are beautiful."
            )
        },
        {
            "name": "Empty text",
            "text": "",
            "expected": ""
        },
        {
            "name": "Whitespace-only text",
            "text": "   \n\t",
            "expected": "   \n\t"
        }
    ]

    passed_tests = 0
    failed_tests = 0
    total_entities = 0

    # Process each integration test
    for index, test_case in enumerate(
        test_cases,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"Test {index}: {test_case['name']}"
        )

        print("=" * 70)

        text = test_case["text"]
        expected = test_case["expected"]

        print("\nOriginal text:")
        print(repr(text))

        try:

            # Execute NER and redaction
            start_time = time.perf_counter()

            entities = detect_entities(text)

            redacted = redact_text(text, entities)

            processing_time = (
                time.perf_counter() - start_time
            )

            print("\nDetected entities:")

            if not entities:
                print("No entities detected.")

            for entity in entities:

                print(
                    f"{entity['text']} | "
                    f"{entity['label']} | "
                    f"{entity['confidence']:.2%} | "
                    f"Offsets: {entity['start']}-"
                    f"{entity['end']}"
                )

                # Verify original character offsets
                assert entity["text"] == text[
                    entity["start"]:entity["end"]
                ], "Incorrect entity offsets."

            print("\nRedacted text:")
            print(repr(redacted))

            # Independently verify the redaction output
            reference_output = build_expected_redaction(
                text,
                entities
            )

            assert redacted == reference_output, (
                "Redaction output does not match "
                "the detected entity positions."
            )

            # Verify predefined expected output
            # when a strict expectation is available
            if expected is not None:

                assert redacted == expected, (
                    f"Expected: {expected!r}\n"
                    f"Actual: {redacted!r}"
                )

                print(
                    "\nExpected output verification: PASSED"
                )

            else:

                print(
                    "\nPrediction quality: "
                    "REQUIRES MANUAL REVIEW"
                )

            total_entities += len(entities)
            passed_tests += 1

            print(f"Entities detected: {len(entities)}")

            print(
                f"Processing time: "
                f"{processing_time:.4f} seconds"
            )

            print("Integration result: PASSED")

        except Exception as error:

            failed_tests += 1

            print("\nIntegration result: FAILED")
            print(f"Error: {error}")

    # Display the final integration test summary
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)

    print(f"Total test cases: {len(test_cases)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total detected entities: {total_entities}")

    print("\nModel cache information:")
    print(load_ner_model.cache_info())

    if failed_tests == 0:

        print(
            "\nAll NER and redaction integration "
            "checks PASSED!"
        )

    else:

        raise AssertionError(
            f"{failed_tests} integration test(s) failed."
        )


if __name__ == "__main__":
    main()
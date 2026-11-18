"""
Unit tests for the automatic text redaction module.

Tests normal inputs, edge cases and invalid entity
information independently of the BERT model.
"""

import unittest

from src.redaction import redact_text


def create_entity(text, label, start, end):
    """
    Create a mock entity dictionary for testing.
    """

    return {
        "text": text,
        "label": label,
        "confidence": 0.99,
        "start": start,
        "end": end
    }


class TestRedaction(unittest.TestCase):

    def test_repeated_entities(self):
        """Redact every detected occurrence."""

        text = "John met John, and John left."

        entities = [
            create_entity("John", "PER", 0, 4),
            create_entity("John", "PER", 9, 13),
            create_entity("John", "PER", 19, 23)
        ]

        expected = (
            "[PERSON] met [PERSON], "
            "and [PERSON] left."
        )

        self.assertEqual(
            redact_text(text, entities),
            expected
        )

    def test_selective_occurrence(self):
        """Redact only the supplied occurrence."""

        text = "John met John."

        entities = [
            create_entity("John", "PER", 9, 13)
        ]

        expected = "John met [PERSON]."

        self.assertEqual(
            redact_text(text, entities),
            expected
        )

    def test_punctuation_preservation(self):
        """Preserve surrounding punctuation."""

        text = "Hello, John! Welcome to London."

        entities = [
            create_entity("John", "PER", 7, 11),
            create_entity("London", "LOC", 24, 30)
        ]

        expected = (
            "Hello, [PERSON]! Welcome to [LOCATION]."
        )

        self.assertEqual(
            redact_text(text, entities),
            expected
        )

    def test_whitespace_preservation(self):
        """Preserve spaces, tabs and newlines."""

        text = "  John\n\tLondon  "

        entities = [
            create_entity("John", "PER", 2, 6),
            create_entity("London", "LOC", 8, 14)
        ]

        expected = "  [PERSON]\n\t[LOCATION]  "

        self.assertEqual(
            redact_text(text, entities),
            expected
        )

    def test_adjacent_entities(self):
        """Handle entities with touching boundaries."""

        text = "JohnLondon"

        entities = [
            create_entity("John", "PER", 0, 4),
            create_entity("London", "LOC", 4, 10)
        ]

        self.assertEqual(
            redact_text(text, entities),
            "[PERSON][LOCATION]"
        )

    def test_unicode_characters(self):
        """Support Unicode text and character offsets."""

        text = "José visited Zürich."

        entities = [
            create_entity("José", "PER", 0, 4),
            create_entity("Zürich", "LOC", 13, 19)
        ]

        expected = "[PERSON] visited [LOCATION]."

        self.assertEqual(
            redact_text(text, entities),
            expected
        )

    def test_no_entities(self):
        """Return the original text when no entities exist."""

        text = "The weather is pleasant."

        self.assertEqual(
            redact_text(text, []),
            text
        )

    def test_empty_input(self):
        """Handle an empty string."""

        self.assertEqual(
            redact_text("", []),
            ""
        )

    def test_negative_offset(self):
        """Reject negative character offsets."""

        text = "John"

        entities = [
            create_entity("John", "PER", -1, 4)
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_offset_beyond_text_length(self):
        """Reject offsets outside the original text."""

        text = "John"

        entities = [
            create_entity("John", "PER", 0, 100)
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_overlapping_entities(self):
        """Reject conflicting entity positions."""

        text = "New York University"

        entities = [
            create_entity("New York", "LOC", 0, 8),
            create_entity(
                "New York University", "ORG", 0, 19
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Overlapping entity offsets detected"
        ):
            redact_text(text, entities)

    def test_unsupported_category(self):
        """Reject categories outside PER, LOC and ORG."""

        text = "English"

        entities = [
            create_entity("English", "MISC", 0, 7)
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_incorrect_entity_text(self):
        """Reject text that does not match its offsets."""

        text = "John"

        entities = [
            create_entity("Peter", "PER", 0, 4)
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_missing_dictionary_keys(self):
        """Reject incomplete entity dictionaries."""

        text = "John"

        entities = [
            {
                "text": "John",
                "label": "PER"
            }
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_invalid_offset_type(self):
        """Reject non-integer character offsets."""

        text = "John"

        entities = [
            create_entity("John", "PER", 0.5, 4)
        ]

        with self.assertRaises(ValueError):
            redact_text(text, entities)

    def test_invalid_input_type(self):
        """Reject non-string text and invalid entity lists."""

        with self.assertRaises(TypeError):
            redact_text(123, [])

        with self.assertRaises(TypeError):
            redact_text("John", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
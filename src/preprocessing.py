"""
Data preprocessing utilities for the NER project.

This module tokenizes CoNLL-2003 sentences and aligns
word-level NER labels with BERT subword tokens.
"""

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize a batch of examples and align their NER labels.

    Args:
        examples: Batch containing 'tokens' and 'ner_tags'.
        tokenizer: Hugging Face BERT tokenizer.

    Returns:
        Tokenized inputs with aligned NER labels.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    all_labels = []

    for i, original_labels in enumerate(examples["ner_tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:

            if word_id is None:
                aligned_labels.append(-100)

            elif word_id != previous_word_id:
                aligned_labels.append(original_labels[word_id])

            else:
                aligned_labels.append(-100)

            previous_word_id = word_id

        all_labels.append(aligned_labels)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs
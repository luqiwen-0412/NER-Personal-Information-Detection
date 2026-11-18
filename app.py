
"""
Streamlit application for AI-Based Personal
Information Detection and Text Redaction.

AIROST-Team LowEnergy

Uses the project's own fine-tuned BERT model
for local named entity recognition.
"""

import time
from html import escape

import pandas as pd
import streamlit as st

from src.ner_model import detect_entities
from src.redaction import redact_text


# Configure the browser page
st.set_page_config(
    page_title="NER Privacy Protector",
    page_icon="🛡️",
    layout="wide"
)


# Define the supported entity categories
CATEGORY_NAMES = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Organisation"
}


# Define colours for entity highlighting
CATEGORY_COLOURS = {
    "PER": "#DBEAFE",
    "LOC": "#DCFCE7",
    "ORG": "#F3E8FF"
}


# Define example input texts
EXAMPLES = {
    "General English example": (
        "John Smith works at Microsoft in London."
    ),
    "Malaysian names and locations": (
        "Ahmad bin Abdullah lives in "
        "Johor Bahru, Malaysia."
    ),
    "Malaysian organisation": (
        "Siti Nurhaliza studied at "
        "Universiti Teknologi Malaysia."
    ),
    "No named entities": (
        "The weather is pleasant "
        "and the flowers are beautiful."
    )
}


def initialise_session():
    """
    Initialise session state variables.
    """

    defaults = {
        "input_text": "",
        "analysis_result": None,
        "selected_categories": ["PER", "LOC", "ORG"]
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


def clear_results():
    """
    Remove previous detection results.
    """

    st.session_state.analysis_result = None


def load_example():
    """
    Load the selected example into the text area.
    """

    selected_example = (
        st.session_state.example_choice
    )

    st.session_state.input_text = (
        EXAMPLES[selected_example]
    )

    clear_results()


def clear_input():
    """
    Clear the input text and detection results.
    """

    st.session_state.input_text = ""

    clear_results()


def display_sidebar():
    """
    Display project and model information.
    """

    with st.sidebar:

        st.header("About the System")

        st.write(
            "This application detects named entities "
            "in raw English text using our own "
            "fine-tuned BERT model."
        )

        st.subheader("Supported Categories")

        st.markdown(
            """
            - **PER:** Person
            - **LOC:** Location
            - **ORG:** Organisation
            """
        )

        st.subheader("Model Information")

        st.write("Base model: bert-base-cased")

        st.write("Training dataset: CoNLL-2003")

        st.write("Inference device: CPU")

        st.write("Maximum sequence length: 512 tokens")

        st.subheader("Baseline Performance")

        st.metric("Test F1-score", "91.07%")

        st.metric(
            "Required-category F1-score",
            "92.82%"
        )

        st.caption(
            "Recorded results from the independent "
            "CoNLL-2003 model evaluation."
        )

        st.divider()

        st.caption(
            "Model inference is performed locally "
            "without an external NER API."
        )


def display_summary(entities, processing_time):
    """
    Display detection counts and processing time.
    """

    total_count = len(entities)

    person_count = sum(
        1 for entity in entities
        if entity["label"] == "PER"
    )

    location_count = sum(
        1 for entity in entities
        if entity["label"] == "LOC"
    )

    organisation_count = sum(
        1 for entity in entities
        if entity["label"] == "ORG"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Entities", total_count)

    col2.metric("Person", person_count)

    col3.metric("Location", location_count)

    col4.metric("Organisation", organisation_count)

    st.caption(
        f"Detection processing time: "
        f"{processing_time * 1000:.2f} ms"
    )


def display_highlighted_text(text, entities):
    """
    Highlight detected entities using their
    original character offsets.
    """

    if not entities:

        st.text(text)

        return

    sorted_entities = sorted(
        entities,
        key=lambda entity: entity["start"]
    )

    parts = []
    cursor = 0

    for entity in sorted_entities:

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        # Preserve text before the entity
        parts.append(
            escape(text[cursor:start])
        )

        # Obtain the original entity text
        entity_text = escape(text[start:end])

        colour = CATEGORY_COLOURS[label]

        category = CATEGORY_NAMES[label]

        # Create a coloured entity label
        highlighted_entity = (
            f'<span style="'
            f'background-color: {colour}; '
            f'color: #111827; '
            f'padding: 3px 5px; '
            f'border-radius: 4px; '
            f'font-weight: 600;">'
            f'{entity_text}'
            f' <small>({category})</small>'
            f'</span>'
        )

        parts.append(highlighted_entity)

        cursor = end

    # Preserve remaining original text
    parts.append(
        escape(text[cursor:])
    )

    highlighted_html = "".join(parts)

    # Display the highlighted text
    st.markdown(
        '<div style="'
        'line-height: 2.4; '
        'padding: 16px; '
        'border: 1px solid #D1D5DB; '
        'border-radius: 8px; '
        'white-space: pre-wrap; '
        'overflow-wrap: anywhere;">'
        + highlighted_html
        + '</div>',
        unsafe_allow_html=True
    )

    # Display the category colour legend
    legend = (
        '<span style="'
        'background:#DBEAFE;color:#111827;'
        'padding:4px 8px;'
        'border-radius:4px;">'
        'Person</span> &nbsp; '

        '<span style="'
        'background:#DCFCE7;color:#111827;'
        'padding:4px 8px;'
        'border-radius:4px;">'
        'Location</span> &nbsp; '

        '<span style="'
        'background:#F3E8FF;color:#111827;'
        'padding:4px 8px;'
        'border-radius:4px;">'
        'Organisation</span>'
    )

    st.markdown(
        legend,
        unsafe_allow_html=True
    )


def display_entity_table(entities):
    """
    Display structured named entity predictions.
    """

    if not entities:

        st.info("No named entities detected.")

        return

    table_rows = []

    for entity in entities:

        table_rows.append({
            "Entity": entity["text"],
            "Category": CATEGORY_NAMES[
                entity["label"]
            ],
            "Confidence": (
                f"{entity['confidence']:.2%}"
            ),
            "Start": entity["start"],
            "End": entity["end"]
        })

    dataframe = pd.DataFrame(table_rows)

    st.dataframe(
        dataframe,
        hide_index=True
    )

    st.caption(
        "Character positions use zero-based indexing. "
        "The end position is exclusive."
    )


def perform_detection(text):
    """
    Execute actual BERT inference and save results.
    """

    # Remove results from any previous analysis
    clear_results()

    if not text.strip():

        st.warning(
            "Please enter some text before detection."
        )

        return

    try:

        with st.spinner(
            "Detecting named entities using BERT..."
        ):

            start_time = time.perf_counter()

            # Actual fine-tuned BERT inference
            entities = detect_entities(text)

            # Validate redaction compatibility
            redact_text(text, entities)

            processing_time = (
                time.perf_counter() - start_time
            )

        # Save results only after successful processing
        st.session_state.analysis_result = {
            "text": text,
            "entities": entities,
            "processing_time": processing_time
        }

        st.success(
            "Named entity detection completed."
        )

    except ValueError as error:

        st.error(str(error))

    except FileNotFoundError:

        st.error(
            "The trained BERT model files "
            "could not be found. Please check "
            "the models/final_model directory."
        )

    except Exception as error:

        st.error(
            "An unexpected error occurred during "
            "named entity detection."
        )

        # Display only the error type
        st.caption(
            f"Error type: {type(error).__name__}"
        )


def display_redaction(text, entities):
    """
    Display selective automatic text redaction.
    """

    st.header("5. Automatic Text Redaction")

    selected_categories = st.multiselect(
        "Select the categories to redact:",
        options=["PER", "LOC", "ORG"],
        format_func=lambda label: CATEGORY_NAMES[label],
        key="selected_categories"
    )

    # Keep only entities from selected categories
    selected_entities = [
        entity for entity in entities
        if entity["label"] in selected_categories
    ]

    try:

        # Execute the verified redaction algorithm
        redacted = redact_text(
            text,
            selected_entities
        )

    except (TypeError, ValueError):

        st.error(
            "The redaction output could not be "
            "generated safely because the entity "
            "information was invalid."
        )

        return

    st.caption(
        f"Entities redacted: "
        f"{len(selected_entities)} / {len(entities)}"
    )

    if not selected_categories:

        st.warning(
            "No categories selected. "
            "The output will contain the original text."
        )

    elif not selected_entities:

        st.info(
            "No detected entities match the "
            "selected redaction categories."
        )

    st.code(
        redacted,
        language=None
    )

    st.download_button(
        label="Download Redacted Text",
        data=redacted.encode("utf-8"),
        file_name="redacted_text.txt",
        mime="text/plain"
    )


def main():
    """
    Display and manage the integrated application.
    """

    initialise_session()

    display_sidebar()

    # Application heading
    st.title("🛡️ NER Privacy Protector")

    st.caption(
        "AIROST | Machine Learning / "
        "Artificial Intelligence | Team LowEnergy"
    )

    st.write(
        "AI-Based Personal Information Detection "
        "and Text Redaction System"
    )

    st.info(
        "Powered by our own fine-tuned BERT model. "
        "Named entity detection runs locally on the CPU."
    )

    st.divider()

    # Text input section
    st.header("1. Text Input")

    st.selectbox(
        "Choose an example:",
        options=list(EXAMPLES.keys()),
        key="example_choice"
    )

    col1, col2 = st.columns(2)

    col1.button(
        "Load Example",
        on_click=load_example
    )

    col2.button(
        "Clear Text",
        on_click=clear_input
    )

    user_text = st.text_area(
        "Enter or paste your text:",
        height=180,
        placeholder="Enter your text here...",
        key="input_text",
        on_change=clear_results
    )

    st.caption(
        f"Characters: {len(user_text)}"
    )

    detect_clicked = st.button(
        "Detect Named Entities",
        type="primary"
    )

    if detect_clicked:

        perform_detection(user_text)

    # Retrieve the saved detection results
    result = st.session_state.analysis_result

    # Ensure results match the current input
    if result is not None:

        if result["text"] != user_text:

            clear_results()

            result = None

    # Display results across Streamlit reruns
    if result is not None:

        original_text = result["text"]

        entities = result["entities"]

        processing_time = (
            result["processing_time"]
        )

        st.divider()

        st.header("2. Detection Summary")

        display_summary(
            entities,
            processing_time
        )

        st.divider()

        st.header("3. Highlighted Text")

        display_highlighted_text(
            original_text,
            entities
        )

        st.divider()

        st.header("4. Detected Entities")

        display_entity_table(entities)

        st.divider()

        display_redaction(
            original_text,
            entities
        )

    st.divider()

    # Privacy limitation notice
    st.warning(
        "This system detects person, location and "
        "organisation entities only. Predictions "
        "may contain errors or miss sensitive "
        "information. Always review the output "
        "before sharing or downloading sensitive text."
    )

    st.caption(
        "Developed by Team LowEnergy | "
        "AIROST ML/AI Internship Project"
    )


if __name__ == "__main__":
    main()
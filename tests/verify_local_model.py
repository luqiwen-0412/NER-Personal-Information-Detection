# Local BERT Model Loading Verification

from pathlib import Path
import json
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

# Identify the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the local model directory
MODEL_DIR = PROJECT_ROOT / "models" / "final_model"

# Configure CPU execution
device = torch.device("cpu")

print("Local BERT Model Verification")
print("------------------------------------")

print("Project root:", PROJECT_ROOT)
print("Model directory:", MODEL_DIR)
print("Execution device:", device)

# Verify required model files
required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "model_information.json"
]

print("\nRequired File Verification")
print("------------------------------------")

for filename in required_files:
    file_path = MODEL_DIR / filename

    print(f"{filename}: {file_path.is_file()}")

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {file_path}"
        )

# Record model loading start time
start_time = time.perf_counter()

# Load the saved tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

# Load the fine-tuned BERT model
model, loading_info = (
    AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        output_loading_info=True
    )
)

# Move the model to CPU
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Calculate loading duration
loading_time = time.perf_counter() - start_time

# Read saved model information
with open(
    MODEL_DIR / "model_information.json",
    "r",
    encoding="utf-8"
) as file:
    model_information = json.load(file)

# Display model loading results
print("\nModel Loading Verification")
print("------------------------------------")

print("Missing keys:",
      loading_info.get("missing_keys", []))

print("Unexpected keys:",
      loading_info.get("unexpected_keys", []))

print("Mismatched keys:",
      loading_info.get("mismatched_keys", []))

print("Model architecture:", type(model).__name__)
print("Model type:", model.config.model_type)
print("Number of labels:", model.config.num_labels)
print("Model parameters:", f"{model.num_parameters():,}")
print("Model device:", next(model.parameters()).device)
print("Tokenizer type:", type(tokenizer).__name__)
print("Evaluation mode:", not model.training)

print("\nNER Label Mapping")
print("------------------------------------")

for label_id, label_name in model.config.id2label.items():
    print(f"{label_id}: {label_name}")

# Display previously recorded evaluation results
print("\nSaved Model Performance")
print("------------------------------------")

print(
    f"Validation F1-score: "
    f"{model_information['validation_f1'] * 100:.2f}%"
)

print(
    f"Test F1-score: "
    f"{model_information['test_f1'] * 100:.2f}%"
)

print(
    f"AIROST required-category F1-score: "
    f"{model_information['required_categories_f1'] * 100:.2f}%"
)

print(f"\nModel loading time: {loading_time:.2f} seconds")

# Ensure there are no parameter loading problems
assert not loading_info.get("missing_keys"), \
    "Missing model parameters detected."

assert not loading_info.get("unexpected_keys"), \
    "Unexpected model parameters detected."

assert not loading_info.get("mismatched_keys"), \
    "Mismatched model parameters detected."

assert model.config.num_labels == 9, \
    "Incorrect number of NER labels."

assert model.num_parameters() == model_information["parameter_count"], \
    "Incorrect model parameter count."

print("\nLocal model loading verification PASSED!")
print("Fine-tuned BERT model is ready for CPU inference.")
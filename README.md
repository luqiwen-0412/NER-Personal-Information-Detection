
# NER Privacy Protector

## AI-Based Personal Information Detection and Text Redaction System

**AIROST UTM — Machine Learning / Artificial Intelligence (ML/AI)**

**Team:** LowEnergy

**Team Members:**
- Ng Jun Jie
- Lu Qi Wen

## 1. Project Overview

NER Privacy Protector is an AI-based application developed to identify and classify named entities in text and automatically redact detected information.

The system uses a fine-tuned BERT model trained on the CoNLL-2003 dataset. It detects three categories of named entities:

- **PER (Person):** Names of individuals.
- **LOC (Location):** Names of geographical locations.
- **ORG (Organisation):** Names of companies, institutions and organisations.

Users can enter or paste English text into a Streamlit application, view detected entities and their confidence scores, and generate a redacted version of the original text.

The project demonstrates how Natural Language Processing (NLP) and Named Entity Recognition (NER) can support the identification and masking of potentially sensitive information.

## 2. Project Objectives

The main objectives of this project are:

1. Fine-tune a pretrained BERT model for Named Entity Recognition.
2. Detect and classify Person, Location and Organisation entities.
3. Develop a user-friendly interface for entering and analysing raw text.
4. Implement automatic text redaction using detected entity positions.
5. Evaluate the fine-tuned model using entity-level performance metrics.

## 3. Main Features

- Fine-tuned `bert-base-cased` model for Named Entity Recognition.
- Detection of Person, Location and Organisation entities.
- Local CPU inference without external NER prediction APIs.
- Interactive web interface developed using Streamlit.
- Colour-coded highlighting of detected entities.
- Entity confidence scores and original character offsets.
- Detection summary with entity counts.
- Automatic replacement of detected entities with category labels.
- Selective redaction based on user-selected categories.
- Downloadable redacted text in `.txt` format.
- Example text selection and Clear Text functionality.
- Input validation and error handling.
- Cached model loading for repeated inference requests.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | BERT fine-tuning and inference |
| Hugging Face Datasets | Dataset loading and preprocessing |
| BERT (`bert-base-cased`) | Base transformer model |
| CoNLL-2003 | Training and evaluation dataset |
| Seqeval | NER performance evaluation |
| Streamlit | Web application |
| Pandas | Data processing and table display |
| Matplotlib | Data visualisation |
| Google Colab | GPU-accelerated model training |
| Git and GitHub | Version control and collaboration |

## 5. Model Development

### 5.1 Dataset

The model was fine-tuned using the CoNLL-2003 dataset.

The dataset contains labelled named entities and is divided into three subsets:

| Dataset Split | Number of Examples |
|---|---:|
| Training | 14,041 |
| Validation | 3,250 |
| Testing | 3,453 |

CoNLL-2003 provides BIO-format annotations for Person, Location, Organisation and Miscellaneous entities.

### 5.2 Tokenisation and Preprocessing

The project uses the `bert-base-cased` tokenizer.

Because BERT may divide words into multiple subword tokens, the preprocessing stage aligns the original NER annotations with the tokenised text.

The first subword receives the original word label, while additional subwords and special tokens receive `-100` so that they are ignored during loss calculation.

The reusable preprocessing implementation is located at:

`src/preprocessing.py`

Dataset exploration and training notebooks are stored in:

`notebooks/`

### 5.3 Model Fine-Tuning

The project fine-tuned `bert-base-cased` for token classification using PyTorch and Hugging Face Transformers.

Training was performed using Google Colab with GPU acceleration.

The model retains nine BIO labels:

- O
- B-PER
- I-PER
- B-ORG
- I-ORG
- B-LOC
- I-LOC
- B-MISC
- I-MISC

The application displays and redacts only PER, LOC and ORG entities.

The fine-tuning notebook is available at:

`notebooks/NER_BERT_FineTuning.ipynb`

### 5.4 Model Evaluation

The fine-tuned baseline model achieved the following recorded results:

| Evaluation Metric | Result |
|---|---:|
| Validation F1-score | 94.66% |
| Test F1-score | 91.07% |
| Required-category F1-score (PER, LOC and ORG) | 92.82% |

Additional evaluation reports and visualisations are available in:

`evaluation_results/`

These are model evaluation results. Application test pass rates should not be interpreted as independent NER accuracy measurements.

## 6. Project Structure

```text
NER-Personal-Information-Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── preprocessing.py
│   ├── ner_model.py
│   └── redaction.py
│
├── tests/
│   ├── verify_local_model.py
│   ├── test_local_inference.py
│   ├── test_reusable_model.py
│   ├── test_redaction.py
│   └── test_ner_redaction_integration.py
│
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── Model_Training.ipynb
│   └── NER_BERT_FineTuning.ipynb
│
├── evaluation_results/
│   └── Training and evaluation reports
│
└── models/
    └── final_model/
        └── Fine-tuned BERT model files
```

The `models/` directory and `.venv/` are excluded from Git version control.

## 7. Installation and Setup

These instructions describe the intended local setup on Windows using PowerShell.

### 7.1 Prerequisites

- Python compatible with the project's dependencies.
- Git.
- Internet access for the initial dependency installation.
- Sufficient memory and storage for the approximately 431 MB fine-tuned BERT model.

The project was developed and tested on Windows using VS Code and a Python virtual environment.

### 7.2 Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/luqiwen-0412/NER-Personal-Information-Detection.git

cd NER-Personal-Information-Detection
```

### 7.3 Create a Virtual Environment

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 7.4 Install Dependencies

Install the dependencies recorded in the project:

```powershell
python -m pip install -r requirements.txt
```

**Note:** The project has been verified in its existing development environment. A complete installation test in a fresh environment has not yet been performed, so package compatibility should be checked when reproducing the setup.

### 7.5 Download the Fine-Tuned BERT Model

The trained BERT model is stored separately from GitHub because of its large file size (approximately 431 MB).

Download the trained model files from Google Drive:

**[Download Fine-Tuned BERT Model](https://drive.google.com/drive/folders/19gTq3JUlmT6QVo6ufsy5BlzecO7h80xR?usp=sharing)**

After downloading, place the model files inside:

`models/final_model/`

The directory should contain:

```text
models/
└── final_model/
    ├── config.json
    ├── model.safetensors
    ├── model_information.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

**Important:** Place the files directly inside `models/final_model/`. Do not create an extra nested `final_model` folder.

The application loads the saved model locally using Hugging Face Transformers. It does not rely on an external NER prediction API.

For training details, refer to:

`notebooks/NER_BERT_FineTuning.ipynb`

### 7.6 Launch the Application

Run the following command in PowerShell:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the application in a browser:

```text
http://localhost:8501
```

To stop the application, press `Ctrl + C` in the terminal.

## 8. How to Use the Application

1. Launch the Streamlit application.
2. Enter or paste English text, or load one of the example sentences.
3. Click **Detect Named Entities**.
4. View the detection summary showing the number of entities.
5. Examine the highlighted entities and their categories.
6. Review the detected entity table, including confidence scores and character offsets.
7. Select the categories to redact.
8. View the automatically redacted text.
9. Click **Download Redacted Text** to save the output as a `.txt` file.

### 8.1 Example Input

```text
John Smith works at Microsoft in London.
```

### 8.2 Detected Entities

| Entity | Category |
|---|---|
| John Smith | Person |
| Microsoft | Organisation |
| London | Location |

### 8.3 Automatic Redaction

When all three categories are selected:

```text
[PERSON] works at [ORGANISATION] in [LOCATION].
```

When only Person is selected:

```text
[PERSON] works at Microsoft in London.
```

The redaction operation replaces entity spans using their original character positions.

## 9. Testing

The project includes tests for model loading, NER inference, automatic redaction and system integration.

### 9.1 Redaction Unit Tests

Run:

```powershell
python -m unittest tests.test_redaction -v
```

Previously verified result:

```text
Ran 16 tests

OK
```

### 9.2 NER and Redaction Integration Tests

Run:

```powershell
python -m tests.test_ner_redaction_integration
```

Previously verified results:

```text
Total test cases: 8
Passed: 8
Failed: 0
```

### 9.3 Manual Application Testing

The application was also tested using:

- General English examples.
- Malaysian names, locations and organisations.
- Repeated entity occurrences.
- Multiple paragraphs.
- Accented Unicode characters.
- Ambiguous entity names.
- HTML-like text.
- Empty input.
- Selective redaction.
- Oversized input.
- Downloadable redacted TXT output.

The results demonstrate functionality for the tested examples but do not guarantee that the model will detect every sensitive entity correctly.

## 10. Current Limitations

The current system has several limitations:

1. **Maximum input length:** The application supports up to 512 BERT tokens per inference sequence. Longer input is rejected rather than silently truncated.
2. **Limited entity categories:** Only Person, Location and Organisation are displayed and redacted.
3. **Model prediction errors:** The model may miss entities, assign incorrect labels or return incorrect boundaries.
4. **Context-dependent predictions:** Ambiguous words may receive different classifications depending on the sentence context.
5. **English-focused training data:** The baseline model was trained using CoNLL-2003. Comprehensive multilingual and Malaysian-specific performance has not been established.
6. **No file input:** The application accepts typed or pasted text. TXT and PDF file uploading is not implemented.
7. **Incomplete anonymisation risk:** Sensitive information that the model does not detect may remain in the redacted output.

Users should always review the output before sharing sensitive text.

## 11. Future Improvements

Possible future enhancements include:

- Processing long documents using token-based chunking.
- Supporting TXT and PDF document uploads.
- Creating an additional labelled Malaysian NER dataset.
- Fine-tuning and evaluating a Malaysian-adapted model.
- Expanding detection to additional personal information categories.
- Improving the recognition of ambiguous entity names.
- Conducting larger-scale domain-specific evaluation.

## 12. Team Information

**Team:** LowEnergy

**Department:** Machine Learning / Artificial Intelligence (ML/AI)

**Organisation:** AIROST UTM Club

**Team Members:**
- Ng Jun Jie
- Lu Qi Wen

## 13. References
1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* https://arxiv.org/abs/1810.04805
2. Tjong Kim Sang, E. F., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.* https://aclanthology.org/W03-0419/
3. Hugging Face Transformers Documentation. https://huggingface.co/docs/transformers/
4. Hugging Face Datasets Documentation. https://huggingface.co/docs/datasets/
5. PyTorch Documentation. https://pytorch.org/docs/stable/index.html
6. Streamlit Documentation. https://docs.streamlit.io/
7. Hugging Face BERT Base Cased Model. https://huggingface.co/google-bert/bert-base-cased
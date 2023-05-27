# SMGPT

Ask questions to your documents without leaving any data footprint, using the power of large language models. Store documents and vector embeddings locally, then query them offline using a local PyTorch LLM model.

## How it works

- Ingest your documents using `ingest.py`
- This will:
  - Parse the documents 
  - Generate sentence embeddings using Sentence Transformers
  - Store the embeddings in a local vector database using DuckDB
- Make queries to your documents using `SMGPT.py`
- This will:
  - Encode your question using the local LLM model
  - Retrieve relevant documents from the local vector database
  - Generate an answer based on the documents and your question

## Benefits

- All document parsing, embedding generation and querying is done locally
- No data leaves your system at any point
- Complete privacy and data ownership

## Requirements

- Python 3.8+
- A C++ compiler (Visual Studio on Windows, GCC on Linux, Xcode on Mac)
- Model requirements:
  - GPT4All-J or LlamaCpp (local PyTorch LLM models)
  - Sentence Transformers (for embeddings generation)

## Usage

1. Place your documents in the `source_documents` folder
2. Install requirements `pip install -r requirements.txt` 
3. Download a GPT4All-J model and place it in the `models` folder
4. Run `ingest.py` to parse and embed your documents
5. Run `SMGPT.py` and enter queries to generate answers

## Limitations

- Lower performance compared to cloud solutions
- Higher system requirements 
- Meant as a proof-of-concept, not production-ready

## Disclaimer

This is a test project to explore the feasibility of fully private and offline document querying. It is not meant for production use and has not been optimized for performance, accuracy, or stability. Use at your own risk.
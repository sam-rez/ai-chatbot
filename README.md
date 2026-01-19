# AI Chatbot with Retrieval-Augmented Generation (RAG)

## Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot**.
It answers questions by **retrieving relevant information from pre-indexed documents** and using a large language model (LLM) to generate answers **strictly from that retrieved context**.

The goal of this project is to demonstrate how modern AI applications separate:

* **Knowledge storage & retrieval**
* **Language understanding & generation**

rather than relying on the LLM’s internal training data alone.

---

## Architecture

The application is split into two phases:

### 1. Ingestion (Offline)

Handled by `ingest.py`.

**What it does:**

* Loads raw text documents
* Splits them into manageable chunks
* Converts each chunk into a vector embedding
* Stores embeddings in a FAISS index on disk

**Why this is offline:**

* Embedding is slow and costs money
* Documents rarely change
* The index can be reused across many queries

**Artifacts created:**

```
index/
├── index.faiss   # Vector index for fast similarity search
└── index.pkl     # Mapping from vectors to original text + metadata
```

---

### 2. Querying (Runtime)

Handled by `chat.py`.

**What it does:**

1. Accepts a user question
2. Converts the question into an embedding
3. Retrieves the most relevant document chunks from the FAISS index
4. Sends the retrieved chunks + the question to the LLM
5. Returns an answer constrained to the retrieved context

This separation ensures the model:

* Uses **external, updatable knowledge**
* Does **not hallucinate** answers
* Can explain *where* information came from

---

## Why This Is a RAG Application

This project qualifies as a RAG system because:

* Knowledge lives **outside the model**
* Relevant context is **retrieved before generation**
* The LLM does not rely on its training data alone
* Updating knowledge requires **re-ingesting documents**, not retraining a model

In short:

> **Retrieval happens first, generation happens second.**

---

## Key Design Decisions

* **Vector search (FAISS)** is used for semantic retrieval instead of keyword search
* **Embeddings are precomputed** to reduce latency and cost
* **Temperature is set to 0** to encourage deterministic, factual answers
* The LLM is explicitly instructed to say *“I don’t know”* when information is missing

---

## What This Project Demonstrates

* How embeddings represent meaning numerically
* How similarity search enables semantic retrieval
* How LLMs act as reasoning layers over retrieved data
* Why RAG is preferred over fine-tuning for most applications
* How to debug RAG systems by inspecting retrieved chunks

---

## Limitations

* No conversation memory (each question is stateless)
* No web UI (CLI only)
* No caching of responses or embeddings
* Assumes documents are in English

These are intentional to keep the core architecture clear.

---

## Possible Improvements

* Add citations directly into answers
* Wrap the chatbot in a FastAPI service
* Add a simple frontend
* Implement query rewriting to improve retrieval
* Add caching for repeated questions
* Swap FAISS or the LLM provider to demonstrate modularity

---

## Setup

### Prerequisites

* Python **3.10+**
* An OpenAI API key
* [`uv`](https://github.com/astral-sh/uv) installed

---

### Create the virtual environment

From the project root:

```bash
uv venv
```

This creates a local virtual environment in `.venv/`.

---

### Install dependencies

```bash
uv pip install \
  langchain \
  langchain-community \
  langchain-openai \
  langchain-text-splitters \
  faiss-cpu
```

---

### Set the OpenAI API key

#### Bash (Git Bash / WSL)

```bash
export OPENAI_API_KEY="sk-REPLACE_ME"
```

(Optional: add this to `~/.bashrc` to persist it.)

#### PowerShell

```powershell
$env:OPENAI_API_KEY="sk-REPLACE_ME"
```

## Running the Project

### Ingest documents

```bash
uv run python ingest.py
```

This builds the FAISS index used for retrieval.

---

### Start the chatbot

```bash
uv run python chat.py
```

---

## Takeaway

This project illustrates a core principle of AI engineering:

> **LLMs are not databases.
> They are reasoning engines layered on top of retrieved knowledge.**

Understanding and implementing this separation is foundational for building reliable, production-grade AI systems.

# AI Chatbot with Retrieval‑Augmented Generation (RAG)

## Overview

This project is a **Retrieval‑Augmented Generation (RAG) chatbot** built to demonstrate **production‑style AI system design**, not just prompt usage.

It answers questions by **retrieving relevant information from pre‑indexed documents** and using a large language model (LLM) to generate answers **strictly from that retrieved context**.

The goal is to show how modern AI applications cleanly separate:

* **Knowledge storage & retrieval**
* **Language understanding & generation**
* **Application interfaces (API / CLI)**

rather than relying on the LLM’s internal training data alone.

---

## High‑Level Architecture

The system is intentionally split into **offline** and **online** components.

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│  Documents │──▶──▶│ Vector Index │──▶──▶│   RAG Engine │──▶ Answer
└────────────┘      └──────────────┘      └──────────────┘
                                           ▲          ▲
                                           │          │
                                       CLI (chat.py)  │
                                                      │
                                               API (app.py)
```

This separation mirrors real production AI services.

---

## Components

### 1. Ingestion (Offline)

Handled by **`ingest.py`**.

**What it does:**

* Loads raw text documents from `docs/`
* Splits them into manageable chunks
* Converts each chunk into a vector embedding
* Persists the embeddings to a vector index on disk

**Why this is offline:**

* Embedding is slow and costs money
* Documents rarely change
* The index can be reused across many queries

**Artifacts created:**

```
index/
├── index.faiss / data files
└── metadata mappings
```

> This index functions as the chatbot’s **external knowledge base**.

---

### 2. RAG Engine (Core Logic)

Handled by **`rag.py`**.

This file contains **all AI‑specific logic**, including:

* Loading embeddings and the vector store
* Retrieving relevant chunks for a query
* Formatting retrieval context
* Calling the LLM
* Returning structured results (answer + citations)

**Key rule:**

> `rag.py` has **no awareness of HTTP or the CLI**.

This makes it reusable, testable, and easy to extend with guardrails, evaluation, or observability later.

---

### 3. Querying Interfaces (Runtime)

#### A. HTTP API — `app.py`

A **FastAPI** application that exposes the chatbot as a service.

**Endpoints:**

* `POST /chat` — main business endpoint

  * Accepts a question
  * Calls the RAG engine
  * Returns a grounded answer with citations

* `GET /health` — monitoring endpoint

  * Confirms the service started successfully
  * Safe for uptime checks and orchestration systems

The API layer is intentionally thin and delegates all logic to `rag.py`.

---

#### B. CLI — `chat.py`

A simple command‑line interface for local development and testing.

**Purpose:**

* Fast iteration without HTTP overhead
* Manual inspection of answers and retrieval behavior

Both the API and CLI call **the same RAG engine**, ensuring a single source of truth.

---

## Why This Is a RAG Application

This project qualifies as a RAG system because:

* Knowledge lives **outside the model**
* Relevant context is **retrieved before generation**
* The LLM is constrained to the retrieved content
* Updating knowledge requires **re‑ingesting documents**, not retraining

In short:

> **Retrieval happens first. Generation happens second.**

---

## Key Design Decisions

* **Vector search** is used for semantic retrieval instead of keyword search
* **Embeddings are precomputed** to reduce latency and cost
* **Temperature = 0** to encourage deterministic, factual responses
* The model is instructed to explicitly say *“I don’t know”* when information is missing
* Core logic is isolated from interfaces (API / CLI)
* A health endpoint is provided for operational monitoring

---

## What This Project Demonstrates

* How embeddings represent meaning numerically
* How similarity search enables semantic retrieval
* How LLMs act as reasoning layers over retrieved data
* How to structure an AI service like a backend system
* Why RAG is preferred over fine‑tuning for most applications
* How to debug RAG systems by inspecting retrieved chunks

---

## Current Limitations (Intentional)

* No conversation memory (each request is stateless)
* No frontend UI
* No retrieval confidence scoring yet
* No caching or evaluation harness

These are intentionally omitted to keep the core architecture clear.

---

## Possible Improvements

* Add retrieval thresholds and confidence scores
* Skip LLM calls when retrieval is weak
* Add structured observability (latency, request IDs)
* Add automated evaluation with golden Q&A sets
* Introduce streaming responses
* Swap vector stores or LLM providers to demonstrate modularity

---

## Setup

### Prerequisites

* Python **3.11+**
* An OpenAI API key
* [`uv`](https://github.com/astral-sh/uv)

---

### Install dependencies

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml` using the locked versions in `uv.lock`.

---

### Set the OpenAI API key

#### Git Bash / WSL

```bash
export OPENAI_API_KEY="sk‑REPLACE_ME"
```

#### PowerShell

```powershell
$env:OPENAI_API_KEY="sk‑REPLACE_ME"
```

---

## Running the Project

### Build the vector index

```bash
uv run python ingest.py
```

---

### Start the API server

```bash
uv run uvicorn app:app --reload
```

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Run the CLI

```bash
uv run python chat.py
```

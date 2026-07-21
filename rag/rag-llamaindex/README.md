# rag-llamaindex

A **RAG (Retrieval-Augmented Generation)** system built with [LlamaIndex](https://www.llamaindex.ai/) that lets you chat with a bot that answers questions grounded in a real project's documentation — instead of relying only on the model's general knowledge.

The chat bot (the Agent) combines two sources of information and decides, per question, which one is relevant:

1. **Semantic (vector) search** over the full documentation, indexed in [Pinecone](https://www.pinecone.io/).
2. **Structured queries** over data extracted from the docs ahead of time — Rules, Warnings and Decisions — stored as local JSONL files.

## Project purpose

The project demonstrates building a Workflow-based Agent (using `llama_index.core.workflow`) that, for every incoming question, performs:

- **Validation** of the input (non-empty, valid length, etc.).
- **Automatic routing** between semantic search and a structured query, using an LLM selector that picks the right tool.
- **Retrieval** from Pinecone or the local JSONL store, with a fallback to a broader query if nothing is found.
- **Confidence scoring** of the retrieved results, warning the user when confidence is low.
- **Answer synthesis** by the LLM, strictly from the retrieved context (to avoid hallucination).
- **Final response formatting**, including a match-quality score.

The chat interface itself is a browser-based [Gradio](https://www.gradio.app/) app.

See [workflow-diagram.html](workflow-diagram.html) for a visual diagram of the workflow (open it in a browser).

## Project structure

| File / folder | Purpose |
|---|---|
| [rag_workflow.py](rag_workflow.py) | 
**Main entry point.** 
The full Workflow (routing, retrieval, confidence scoring, synthesis) plus the Gradio UI. |
| [prepare.py](prepare.py) | Data preparation script: loads documents, builds the Pinecone index, and extracts structured Rules / Warnings / Decisions via an LLM. **Run this before using `rag_workflow.py` for the first time.** |
| [objects.py](objects.py) | Pydantic models for the structured data: `rules`, `warnings`, `decisions`, `Source` and `StructuredQuery`. |
| [file_storage.py](file_storage.py) | Simple storage layer over JSONL files (`output/*.jsonl`) — save, load and filter the structured data. |
| [agent.py](agent.py) / [workflow.py](workflow.py) | Earlier/simpler versions of the same idea (no routing, no confidence scoring). Kept for reference — use `rag_workflow.py` for normal use. |
| `kiro-steering/`, `claude/` | Sample documentation folders that are indexed and serve as the Agent's knowledge source. |
| `output/` | Extraction output: `rules.jsonl`, `warnings.jsonl`, `decisions.jsonl`. |

> The indexed documentation can easily be swapped out: just point `folders` in `prepare.py` at different folders in your own project.

## Prerequisites

- Python 3.14+
- Accounts (free tier is enough to experiment) with:
  - [OpenAI](https://platform.openai.com/) — for the LLM (`gpt-4o` / `gpt-4o-mini`).
  - [Cohere](https://cohere.com/) — for the embedding model.
  - [Pinecone](https://www.pinecone.io/) — for the vector store. You need to create an index named `rag-llamaindex` beforehand.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

(You can also use `pip install -e .` if you prefer plain pip, via `pyproject.toml`.)

Create a `.env` file in the project root with the following keys:

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
PINECONE_API_KEY=...
```

## How to run

### Step 1 — Prepare the data (one-time, or whenever the docs change)

Indexes the documents into Pinecone and extracts Rules/Warnings/Decisions into JSONL files under `output/`:

```bash
uv run python prepare.py
```

### Step 2 — Run the Agent

```bash
uv run python rag_workflow.py
```

This starts a local Gradio server (the URL is printed in the terminal, usually `http://127.0.0.1:7860`) with a chat interface for talking to the Agent.

## Example questions the Agent can answer

**Open-ended / explanatory questions** (routed to semantic search over the full documentation):

- "What is the tech stack used in this project?"
- "How is the client folder structured?"
- "What AWS services does the game use?"
- "Explain the frontend systems architecture."

**Factual / list-like / time-based questions** (routed to a structured query over Rules / Warnings / Decisions):

- "List all high severity warnings."
- "What rules apply to the UI / Game Mechanics scope?"
- "What decisions were made about the frontend architecture?"
- "What warnings were added in the last week?"

When the Agent can't find enough relevant context, or its confidence in the answer is low, it will say so explicitly instead of making something up — including a match-quality score (`Best match score`) at the end of every answer.

import asyncio
import json
import os
from datetime import datetime, timezone
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.schema import NodeWithScore, TextNode
from typing import Optional
from file_storage import FileStorage
from objects import StructuredQuery
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import PydanticSingleSelector



# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

for name, val in [
    ("PINECONE_API_KEY", PINECONE_API_KEY),
    ("COHERE_API_KEY",   COHERE_API_KEY),
    ("OPENAI_API_KEY",   OPENAI_API_KEY),
]:
    if not val:
        raise ValueError(f"{name} is missing from the environment")


# ─────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────
# Nodes whose similarity score is below MIN_NODE_SCORE are discarded.
# If the best remaining score is below CONFIDENCE_THRESHOLD the workflow
# treats the retrieval as "low confidence" and uses the fallback path.
MIN_NODE_SCORE       = 0.01   # per-node filter
CONFIDENCE_THRESHOLD = 0.55   # top-score gate
MIN_QUERY_LEN        = 3
MAX_QUERY_LEN        = 500
NO_ANSWER_SIGNAL     = "i don't have enough information"


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
TEXT_QA_TEMPLATE = PromptTemplate(
    """You are a senior developer and the architect of the project.
Answer clearly using ONLY the context below.
If the context mentions a system-design decision, name it and explain it.
If the context does not contain the answer, reply with exactly:
"I don't have enough information"

Context:
{context_str}

Question:
{query_str}

Answer:"""
)


# ─────────────────────────────────────────────
# Events  (one per workflow step boundary)
# ─────────────────────────────────────────────

class QueryValidatedEvent(Event):
    """Step 1 → Step 2: input passed basic checks."""
    message: str

class RouterEvent(Event):
    """Step 2 → Step 3: input passed basic checks."""
    tool : RetrieverTool

class NodesRetrievedEvent(Event):
    """Step 3 → Step 4: raw retrieval results."""
    message: str
    nodes: list[NodeWithScore]

class NodesScoredEvent(Event):
    """Step 4 → Step 5: filtered, confidence-scored nodes."""
    message:    str
    nodes:      list[NodeWithScore]
    top_score:  float
    confident:  bool

class ResponseReadyEvent(Event):
    """Step 5 → Step 6: raw LLM answer before final checks."""
    answer:    str
    top_score: float
    confident: bool

class StructuredQueryEvent(Event):
    """Step 2 → Step 3b: router decided the question needs the structured data store."""
    message: str

# ─────────────────────────────────────────────
# Infrastructure helpers
# ─────────────────────────────────────────────

def load_index() -> VectorStoreIndex:
    pc             = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("rag-llamaindex")
    vector_store   = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="kiro-steering",
    )
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def _no_answer(text: str) -> bool:
    """True when the LLM admitted it could not answer."""
    return NO_ANSWER_SIGNAL in text.lower()


def _broaden_query(query: str) -> str:
    """
    Very simple broadening: drop stop-words and keep the first 4 tokens.
    Replace with a proper NLP approach if needed.
    """
    stop = {"what","is","are","the","a","an","of","in","on","at","for","to","do","does","how","why","when","where","who"}
    tokens = [t for t in query.lower().split() if t not in stop]
    return " ".join(tokens[:4]) if tokens else query


# ─────────────────────────────────────────────
# Workflow
# ─────────────────────────────────────────────

class RAGWorkflow(Workflow):

    # ── Step 1 ── Validate input ──────────────────────────────────────
    @step
    async def validate_input(self, ev: StartEvent) -> QueryValidatedEvent | StopEvent:
        """
        Validates the raw user query before touching any external service.

        Checks:
        - Not empty / only whitespace
        - Meets minimum length (avoids meaningless single-char queries)
        - Does not exceed maximum length
        """
        message: str = (ev.get("message") or "").strip()

        if not message:
            return StopEvent(result="❌ Please provide a question.")

        if len(message) < MIN_QUERY_LEN:
            return StopEvent(result=f"❌ Question too short (minimum {MIN_QUERY_LEN} characters).")

        if len(message) > MAX_QUERY_LEN:
            return StopEvent(result=f"❌ Question too long (maximum {MAX_QUERY_LEN} characters).")

        return QueryValidatedEvent(message=message)
    
    @step
    async def route(self, ev: QueryValidatedEvent) -> RouterEvent | StructuredQueryEvent:
        """
        Uses a LlamaIndex selector to decide, per question, whether to answer
        from the structured data store (list_tool) or from semantic search
        over the indexed docs (vector_tool).

        Checks:
        - Selector picks exactly one of the two tool descriptions
        - index 0 → vector_tool (semantic search)
        - index 1 → list_tool   (structured rules/warnings/decisions lookup)
        """
        choices = [vector_tool.metadata, list_tool.metadata]
        result = selector.select(choices, ev.message)
        selection = result.selections[0]

        print(f"[ROUTER] query={ev.message!r}")
        print(f"[ROUTER] chosen index={selection.index} reason={selection.reason!r}")

        if selection.index == 1:
            print("[ROUTER] -> structured data lookup")
            return StructuredQueryEvent(message=ev.message)

        print("[ROUTER] -> semantic search")
        return RouterEvent(message=ev.message, tool=vector_tool)


    # ── Step 3a ── Structured lookup ──────────────────────────────────
    @step
    async def structured_retrieve(self, ev: StructuredQueryEvent) -> ResponseReadyEvent | StopEvent:
        """
        Handles questions routed to the structured data store.

        - Asks the LLM to translate the question into a StructuredQuery
          (data_type + field filters + optional since-timestamp) matching
          the schema defined in objects.py.
        - Runs that query against FileStorage (the JSONL structured store).
        - Sends the matching items back to the LLM to write the final answer.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        query_llm = llm.as_structured_llm(output_cls=StructuredQuery)
        prompt = (
            f"The current date/time is {now_iso}.\n"
            "Translate the user's question into a structured query over the project's "
            "extracted data.\n"
            "data_type must be exactly one of: rules, warnings, decisions.\n"
            "Only set fields that are exact-match filters relevant to the chosen data_type "
            "and that you are confident about; leave the rest null:\n"
            "- rules: use scope\n"
            "- warnings: use area, severity\n"
            "- name can be set for any data_type only if the question names a specific "
            "item exactly.\n"
            "If the question asks for a full or complete list ('all', 'every', 'list all') "
            "and does not mention a specific category, leave every filter null so the whole "
            "list is returned - do not invent a filter from a loose paraphrase of the question.\n"
            "since is an ISO-8601 timestamp string. Only set it when the question is "
            "explicitly time-based (e.g. 'recent', 'in the last week', 'this month'); compute "
            "it as an absolute timestamp relative to the current date/time above.\n"
            f"Question: {ev.message}"
        )
        query: StructuredQuery = query_llm.complete(prompt).raw
        print(f"[STRUCTURED] generated query: {query.model_dump()}")

        allowed_fields = {
            "rules": ("name", "scope"),
            "warnings": ("name", "area", "severity"),
            "decisions": ("name",),
        }[query.data_type]
        filters = {
            field: getattr(query, field)
            for field in allowed_fields
            if getattr(query, field) is not None
        }
        print(f"[STRUCTURED] applied filters: {filters or 'none'}")

        items = storage.load(query.data_type, filters=filters or None, since=query.since)

        if not items and filters:
            # ── Fallback: the LLM guessed a filter value (e.g. scope="RTL") that
            # doesn't literally match the stored vocabulary. Retry unfiltered
            # (keeping only the time filter) and let the answer LLM judge
            # relevance from full item text instead of silently returning nothing.
            print(f"[STRUCTURED] no hits with filters {filters}, retrying without field filters")
            items = storage.load(query.data_type, filters=None, since=query.since)

        items.sort(key=lambda i: i.get("last_updated", ""), reverse=True)
        print(f"[STRUCTURED] fetched {len(items)} {query.data_type} item(s)")

        if not items:
            return StopEvent(
                result=(
                    f"🔍 No {query.data_type} matched your question "
                    f"(filters: {filters or 'none'})."
                )
            )

        context = json.dumps(items, indent=2, default=str)
        answer_prompt = (
            "Answer the user's question using ONLY the structured data below.\n"
            "If the question asks for a full or complete list, include EVERY item below "
            "that is relevant - do not summarize down to just one.\n"
            "Be specific and mention item names. Items are ordered most-recent-first "
            "(by last_updated). Only use recency to break a tie when two or more items "
            "clearly describe the SAME name/topic but contradict each other - in that case "
            "say the most recent one is the current/valid one. Do not drop unrelated items "
            "just because a newer item exists.\n\n"
            f"Data:\n{context}\n\nQuestion: {ev.message}\n\nAnswer:"
        )
        response = llm.complete(answer_prompt)
        print("[STRUCTURED] final answer generated")

        return ResponseReadyEvent(answer=str(response).strip(), top_score=1.0, confident=True)

    # ── Step 3b ── Retrieve ────────────────────────────────────────────
    @step
    async def retrieve(self, ev: RouterEvent) -> NodesRetrievedEvent | StopEvent:
        """
        Runs vector search in Pinecone.

        Checks:
        - retriever returns at least one node
        - If nothing found: try once with a broader query;
          if still nothing, return a friendly "not found" message.
        """
        retriever = ev.tool.retriever
        nodes = retriever.retrieve(ev.message)
        print(f"[RETRIEVE] semantic search returned {len(nodes)} node(s)")

        if not nodes:
            # ── Fallback A: retry with a simpler query ──────────────
            broad  = _broaden_query(ev.message)
            print(f"[RETRIEVE] no hits, retrying with broadened query: {broad!r}")
            nodes  = retriever.retrieve(broad) if broad != ev.message else []
            print(f"[RETRIEVE] broadened search returned {len(nodes)} node(s)")

        if not nodes:
            return StopEvent(
                result=(
                    "🔍 I couldn't find relevant documentation for your question.\n"
                    "The indexed files may not cover this topic yet."
                )
            )

        return NodesRetrievedEvent(message=ev.message, nodes=nodes)

    # ── Step 3 ── Postprocess / score ─────────────────────────────────
    @step
    async def postprocess(self, ev: NodesRetrievedEvent) -> NodesScoredEvent | StopEvent:
        """
        Filters nodes by per-node score and evaluates overall confidence.

        Checks:
        - Discards nodes below MIN_NODE_SCORE
        - If all nodes are discarded → "not found"
        - Computes top_score and sets confident=False when below CONFIDENCE_THRESHOLD
          (workflow continues but the final answer will carry a warning)
        """
        scored_nodes = [n for n in ev.nodes if (n.score or 0) >= MIN_NODE_SCORE]

        if not scored_nodes:
            return StopEvent(
                result=(
                    "🔍 Retrieved results were not relevant enough to answer your question.\n"
                    f"Best similarity score was {max((n.score or 0) for n in ev.nodes):.2f} "
                    f"(threshold: {MIN_NODE_SCORE})."
                )
            )

        top_score = max(n.score or 0 for n in scored_nodes)
        confident = top_score >= CONFIDENCE_THRESHOLD

        return NodesScoredEvent(
            message=ev.message,
            nodes=scored_nodes,
            top_score=top_score,
            confident=confident,
        )

    # ── Step 4 ── Synthesize ──────────────────────────────────────────
    @step
    async def synthesize(self, ev: NodesScoredEvent) -> ResponseReadyEvent | StopEvent:
        """
        Sends filtered context + query to the LLM.

        Checks:
        - Response is not empty
        - If the LLM itself says it doesn't know (NO_ANSWER_SIGNAL),
          flag it so Step 5 can add a disclaimer rather than silently returning.
        """
        response = response_synthesizer.synthesize(ev.message, nodes=ev.nodes)
        answer   = str(response).strip()

        if not answer:
            return StopEvent(result="⚠️ The model returned an empty response. Please try again.")

        return ResponseReadyEvent(
            answer=answer,
            top_score=ev.top_score,
            confident=ev.confident,
        )

    # ── Step 5 ── Format & return ─────────────────────────────────────
    @step
    async def return_response(self, ev: ResponseReadyEvent) -> StopEvent:
        """
        Assembles the final user-facing message.

        Adds:
        - A low-confidence disclaimer when confident=False
        - An "insufficient information" notice when the LLM admitted it couldn't answer
        - A source-quality footer showing the top similarity score
        """
        answer = ev.answer

        # ── Confidence disclaimer ───────────────────────────────────
        if not ev.confident:
            answer = (
                f"⚠️ *Low confidence answer (top score: {ev.top_score:.2f})*\n\n"
                + answer
            )

        # ── LLM "I don't know" signal ───────────────────────────────
        if _no_answer(ev.answer):
            answer = (
                "🔍 The indexed documentation does not contain a clear answer.\n\n"
                + answer
            )

        # ── Source quality footer ───────────────────────────────────
        answer += f"\n\n---\n*Best match score: {ev.top_score:.2f}*"

        return StopEvent(result=answer)


# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

llm = OpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.1,
)
response_synthesizer = get_response_synthesizer(
    llm=llm,
    text_qa_template=TEXT_QA_TEMPLATE,
    response_mode=ResponseMode.COMPACT,
)

index    = load_index()
vector_retriever  = index.as_retriever(similarity_top_k=3)

def _item_to_text(item: dict) -> str:
    """Render one structured item (rule/warning/decision) as plain text for indexing."""
    lines = [f"type: {item.get('_type')}", f"name: {item.get('name')}"]
    for key in ("title", "summary", "decision", "notes", "scope", "area", "message", "severity"):
        if item.get(key):
            lines.append(f"{key}: {item[key]}")
    return "\n".join(lines)

storage = FileStorage(output_dir="output")
structured_items = storage.load_all()
structured_nodes = [TextNode(text=_item_to_text(item), metadata=item) for item in structured_items]
file_retriever = SummaryIndex(nodes=structured_nodes).as_retriever()

# Tools
list_tool = RetrieverTool.from_defaults(
    retriever=file_retriever,
    description=(
        "Useful for factual, list-like, or up-to-date questions about the project's "
        "extracted rules, warnings, and decisions (see objects.py) - e.g. 'list all "
        "high severity warnings', 'what rules apply to the UI', 'what decisions were "
        "made about the DB'."
    ),
)
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description=(
        "Useful for open-ended, explanatory questions about the indexed agentic-coding "
        "tool documentation (kiro-steering folder) - general project/codebase/design context."
    ),
)

selector = PydanticSingleSelector.from_defaults(llm=llm)

workflow  = RAGWorkflow(timeout=60, verbose=True)


# ─────────────────────────────────────────────
# Gradio interface
# ─────────────────────────────────────────────

async def chat(message: str, history: list) -> str:
    try:
        result = await workflow.run(message=message)
        return result.get("result") if isinstance(result, dict) else str(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Unexpected error: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    title="RAG Chat — Event-Driven Workflow",
    description="Ask questions about the indexed agentic-coding tool documentation.",
)

if __name__ == "__main__":
    demo.launch()
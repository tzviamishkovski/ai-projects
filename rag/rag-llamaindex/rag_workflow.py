import asyncio
import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.schema import NodeWithScore
from typing import Optional

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
MIN_NODE_SCORE       = 0.30   # per-node filter
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

class NodesRetrievedEvent(Event):
    """Step 2 → Step 3: raw retrieval results."""
    message: str
    nodes: list[NodeWithScore]

class NodesScoredEvent(Event):
    """Step 3 → Step 4: filtered, confidence-scored nodes."""
    message:    str
    nodes:      list[NodeWithScore]
    top_score:  float
    confident:  bool

class ResponseReadyEvent(Event):
    """Step 4 → Step 5: raw LLM answer before final checks."""
    answer:    str
    top_score: float
    confident: bool


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

    # ── Step 2 ── Retrieve ────────────────────────────────────────────
    @step
    async def retrieve(self, ev: QueryValidatedEvent) -> NodesRetrievedEvent | StopEvent:
        """
        Runs vector search in Pinecone.

        Checks:
        - retriever returns at least one node
        - If nothing found: try once with a broader query;
          if still nothing, return a friendly "not found" message.
        """
        nodes = retriever.retrieve(ev.message)

        if not nodes:
            # ── Fallback A: retry with a simpler query ──────────────
            broad  = _broaden_query(ev.message)
            nodes  = retriever.retrieve(broad) if broad != ev.message else []

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

index    = load_index()
retriever = index.as_retriever(similarity_top_k=5)
workflow  = RAGWorkflow(timeout=60, verbose=True)

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
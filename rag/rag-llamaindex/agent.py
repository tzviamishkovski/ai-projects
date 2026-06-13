import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

#initilize environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from the environment")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is missing from the environment")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from the environment")


TEXT_QA_TEMPLATE = PromptTemplate(
    """You are a helpful assistant.
    Answer the user's question using the context below only.
    If the context does not contain the answer, say that you do not know.

    Context:
    {context_str}

    Question:
    {query_str}

    Answer:"""
    )


def load_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("rag-llamaindex")

    vector_store = PineconeVectorStore(
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

index = load_index()
retriever = index.as_retriever()

llm = OpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.1,
)
response_synthesizer = get_response_synthesizer(
    llm=llm,
    text_qa_template=TEXT_QA_TEMPLATE,
    response_mode=ResponseMode.COMPACT
)

def chat(message, history):
    nodes = retriever.retrieve(message)
    if not nodes:
        return "No relevant context found."

    response = response_synthesizer.synthesize(message, nodes=nodes)
    return str(response)


demo = gr.ChatInterface(
    fn=chat,
    title="RAG Chat",
    description="Ask a question about the indexed documents.",
)


if __name__ == "__main__":
    demo.launch()
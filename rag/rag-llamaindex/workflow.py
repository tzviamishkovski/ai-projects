import asyncio
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
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event

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
    """You like a senior developer and the architct of the project.
    Answer clearly the user's question using the context below only.     
    if the solution use one of system design inform the user and explain it.
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

class RetrieveEvent(Event):
    """Event for passing retrieved nodes"""
    nodes: list
    message: str

class ResponseEvent(Event):
    response: str

class ValidationEvent(Event):
    """Event for validating the input message"""
    message: str

class PostValidationEvent(Event):
    """Event for validating the response"""
    result: str

class RAGWorkflow(Workflow):

    @step
    async def validate(self, ev: StartEvent) -> ValidationEvent:
        """Validate the input message"""
        message = ev.get("message")
        
        # Add your validation logic here
        if not message or len(message.strip()) == 0:
            return StopEvent(result="Please provide a valid question.")
        
        if len(message) > 500:
            return StopEvent(result="Question is too long. Please keep it under 500 characters.")
        
        # Pass to retrieve step
        return ValidationEvent(nodes=None, message=message)
    
    @step
    async def retrieve(self, ev: ValidationEvent) -> RetrieveEvent:
        """Retrieve nodes from the index"""
        message = ev.message
        nodes = retriever.retrieve(message)
        return RetrieveEvent(nodes=nodes, message=message)
    
    @step
    async def synthesize(self, ev: RetrieveEvent) -> ResponseEvent :
        """Synthesize response from nodes"""
        if not ev.nodes:
            return StopEvent(result="No relevant context found.")
        
        response = response_synthesizer.synthesize(
            ev.message, 
            nodes=ev.nodes
        )
        return ResponseEvent(response=str(response))
    
    @step
    async def post_validate(self, ev: ResponseEvent) -> PostValidationEvent:
        """Validate the output message"""
        result = ev.response
        
        # Add your validation logic here
        if not result or len(result.strip()) == 0:
            return StopEvent(result="Please provide a valid question.")
        
        if len(result) > 500:
            return StopEvent(result="Question is too long. Please keep it under 500 characters.")
        
        # Pass to retrieve step
        return PostValidationEvent(nodes=None, result=result)

# Usage
index = load_index()
retriever = index.as_retriever()
workflow = RAGWorkflow(timeout=60, verbose=True)


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
async def chat(message, history):
    try:
        result = await workflow.run(message=message)
        return result.get("result") if isinstance(result, dict) else str(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat,
    title="RAG Chat with Workflow",
    description="Ask a question about the indexed documents.",
)


if __name__ == "__main__":
    demo.launch()
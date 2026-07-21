import json
import os
from dotenv import load_dotenv
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from sqlalchemy import Extract
from objects import rules, warnings, decisions
from file_storage import FileStorage

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

#initilize environment variables
load_dotenv()
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is missing from the environment")  
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from the environment")    


def load_documents(folder_name: str):
    """Load documents from the given folder."""
    reader = SimpleDirectoryReader(input_dir=folder_name)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents from {folder_name}")
    return documents

def build_index(folder_name: str, documents):
    """Build a vector index from documents and store it in Pinecone."""
    # Chunking
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(
        documents=documents, show_progress=True
    )

    # Add folder metadata to each node before indexing.
    for node in nodes:
        node.extra_info["folder_name"] = folder_name

    # Embedding
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_document",
    )

    # Indexing and Saving
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("rag-llamaindex")
    pinecone_vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=folder_name,
    )

    storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)

    return VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

folders = ["kiro-steering", "claude"]
documents_by_folder = {folder: load_documents(folder) for folder in folders}
indexes = {folder: build_index(folder, docs) for folder, docs in documents_by_folder.items()}
index = indexes["kiro-steering"]


#Data Extraction
llm = OpenAI(model="gpt-4o")
rules_sllm = llm.as_structured_llm(output_cls=rules)
warnings_sllm = llm.as_structured_llm(output_cls=warnings)
decisions_sllm = llm.as_structured_llm(output_cls=decisions)

def extract_data(sllm, data_type, document_text):
    """Generic function to extract structured data from document text"""
    try:
        prompt = f"based on the context return a list of {data_type} that implements in the project, the {data_type} should be in the format of the {data_type} object. name is unique, last_updated is the current timestamp although you have a specific time, the source will be the AI helper and the url will be the document  and name you depend on and the original_text will be the sentence text you depend on to determine thid {data_type}, make sure to depend only on the document context: {document_text}"
        response = sllm.complete(prompt)
        print(f"{data_type.capitalize()} extracted:")
        print(json.dumps(response.raw, indent=2, default=str))
        return response.raw
    except Exception as e:
        print(f"Error extracting {data_type}: {e}")
        return None

#------------------local Storage

storage = FileStorage(output_dir="output")

# Process all documents from the first loaded folder.
# If you want to extract from both folders, iterate over `documents_by_folder`.
all_rules = []
all_warnings = []
all_decisions = []

for folder_name, documents in documents_by_folder.items():
    for i, document in enumerate(documents):
        document_text = document.text
        print(f"\n--- Processing document {i+1}/{len(documents)} from {folder_name} ---")

        extractions = {
            "rules":     extract_data(rules_sllm,     "rules",     document_text),
            "warnings":  extract_data(warnings_sllm,  "warnings",  document_text),
            "decisions": extract_data(decisions_sllm, "decisions", document_text),
        }

        for data_type, result in extractions.items():
            if result:
                items = result if isinstance(result, list) else [result]
                storage.save(data_type, items)

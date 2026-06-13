import os
from dotenv import load_dotenv
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.cohere import CohereEmbedding

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

# Loading
reader = SimpleDirectoryReader(input_dir="kiro-steering")
documents = reader.load_data()
print(len(documents))

# Chunking

node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
node_parser=node_parser
#node_md = MarkdownNodeParser()
nodes = node_parser.get_nodes_from_documents(
    documents=documents, show_progress=True
)

# Embedding
embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

# Indexing and Saving
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("rag-llamaindex")
pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="kiro-steering")

storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)

index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


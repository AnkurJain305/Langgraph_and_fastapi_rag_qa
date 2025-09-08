from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Config
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "qa_collection"

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Docker Chroma
_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def _new_vs() -> Chroma:
    return Chroma(
        client=_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=_embeddings,
    )

def ingest_documents(docs: List[Document]) -> int:
    vs = _new_vs()
    vs.add_documents(docs)
    return len(docs)

def get_retriever(k: int = 4):
    vs = _new_vs()
    return vs.as_retriever(search_kwargs={"k": k})

from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


CHROMA_DIR = Path("./chroma_db")

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _new_vs() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=_embeddings)

def ingest_documents(docs: List[Document]) -> int:
    vs = _new_vs()
    vs.add_documents(docs)
    vs.persist()
    return len(docs)

def get_retriever(k: int = 4):
    vs = _new_vs()
    return vs.as_retriever(search_kwargs={"k": k})

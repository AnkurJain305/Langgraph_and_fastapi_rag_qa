from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from services.pdf_service import load_and_split_pdf
from services.vectorstore import ingest_documents

class IngestState(TypedDict, total=False):
    file_path: str
    docs: List[Document]
    n_chunks: int
    error: Optional[str]

def load_pdf_node(state: IngestState) -> IngestState:
    try:
        docs = load_and_split_pdf(state["file_path"])
        return {"docs": docs, "n_chunks": len(docs)}
    except Exception as e:
        return {"error": f"Load/Chunk failed: {e}"}

def embed_store_node(state: IngestState) -> IngestState:
    try:
        docs = state.get("docs", [])
        n = ingest_documents(docs)
        return {"n_chunks": n}
    except Exception as e:
        return {"error": f"Embed/Store failed: {e}"}

ingest_builder = StateGraph(IngestState)
ingest_builder.add_node("load_pdf", load_pdf_node)
ingest_builder.add_node("embed_store", embed_store_node)
ingest_builder.set_entry_point("load_pdf")
ingest_builder.add_edge("load_pdf", "embed_store")
ingest_builder.add_edge("embed_store", END)

ingest_graph = ingest_builder.compile()

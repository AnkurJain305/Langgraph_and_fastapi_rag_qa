from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import os

from vectorstore import ingest_documents, get_retriever

# ---------- Ingest state & graph ----------
class IngestState(TypedDict, total=False):
    file_path: str
    docs: List[Document]
    n_chunks: int
    error: Optional[str]

def load_pdf_node(state: IngestState) -> IngestState:
    try:
        loader = PyPDFLoader(state["file_path"])
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(raw_docs)
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

# ---------- QA state & graph ----------
class QAState(TypedDict, total=False):
    question: str
    retrieved: List[Document]
    answer: str
    db: Any          
    record_id: int
    error: Optional[str]

def retrieve_node(state: QAState) -> QAState:
    try:
        retriever = get_retriever(k=4)
        retrieved = retriever.get_relevant_documents(state["question"])
        return {"retrieved": retrieved}
    except Exception as e:
        return {"error": f"Retrieve failed: {e}"}

def generate_answer_node(state: QAState) -> QAState:
    try:
        os.environ["OPENAI_API_KEY"] = ""  
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        context = "\n\n".join([d.page_content for d in state.get("retrieved", [])])
        prompt = (
            "You are a helpful assistant. Use the context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer clearly and concisely."
        )
        answer = llm.invoke(prompt).content
        return {"answer": answer}
    except Exception as e:
        return {"error": f"LLM failed: {e}"}

def save_to_db_node(state: QAState) -> QAState:
    try:
        db = state.get("db")
        if db is None:
            return {"error": "No DB session provided to save_to_db_node."}
        import crud, schemas
        created = crud.create_qa(db, schemas.QACreate(question=state["question"], answer=state["answer"]))
        return {"record_id": created.id}
    except Exception as e:
        return {"error": f"DB save failed: {e}"}

qa_builder = StateGraph(QAState)
qa_builder.add_node("retrieve", retrieve_node)
qa_builder.add_node("generate", generate_answer_node)
qa_builder.add_node("save", save_to_db_node)
qa_builder.set_entry_point("retrieve")
qa_builder.add_edge("retrieve", "generate")
qa_builder.add_edge("generate", "save")
qa_builder.add_edge("save", END)
qa_graph = qa_builder.compile()

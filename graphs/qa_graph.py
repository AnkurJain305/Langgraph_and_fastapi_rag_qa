from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import crud.crud as crud
import schemas.schemas as schemas

from services.vectorstore import get_retriever


load_dotenv()

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
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
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
            return {"error": "No DB session provided"}
        
        created = crud.create_qa(
            db, schemas.QACreate(question=state["question"], answer=state["answer"])
        )
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

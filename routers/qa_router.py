from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path

from graphs import ingest_graph, qa_graph, IngestState, QAState
from database import get_db
from schemas.schemas import AskRequest, QACreate, QAResponse
import crud.crud as crud

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Upload PDF
@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        f.write(await file.read())

    initial: IngestState = {"file_path": str(dest)}
    result = ingest_graph.invoke(initial)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"message": "Ingested successfully"}


# Ask Question
@router.post("/ask")
async def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    initial: QAState = {"question": request.question, "db": db}
    result = qa_graph.invoke(initial)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "question": request.question,
        "answer": result.get("answer"),
        "record_id": result.get("record_id")
    }


# CRUD Endpoints
@router.post("/qa/", response_model=QAResponse)
def create_qa(qa: QACreate, db: Session = Depends(get_db)):
    return crud.create_qa(db=db, qa=qa)

@router.get("/qa/", response_model=list[QAResponse])
def read_qas(db: Session = Depends(get_db)):
    return crud.get_qas(db=db)

@router.get("/qa/{qa_id}", response_model=QAResponse)
def read_qa(qa_id: int, db: Session = Depends(get_db)):
    db_qa = crud.get_qa(db, qa_id)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return db_qa

@router.put("/qa/{qa_id}", response_model=QAResponse)
def update_qa(qa_id: int, qa: QACreate, db: Session = Depends(get_db)):
    db_qa = crud.update_qa(db, qa_id, qa)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return db_qa

@router.delete("/qa/{qa_id}")
def delete_qa(qa_id: int, db: Session = Depends(get_db)):
    db_qa = crud.delete_qa(db, qa_id)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return {"message": "QA deleted successfully"}

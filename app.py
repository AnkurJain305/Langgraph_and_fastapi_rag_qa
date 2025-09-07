from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from graphs import ingest_graph, qa_graph, IngestState, QAState
from database import get_db
from schemas import AskRequest

router = APIRouter()
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        f.write(await file.read())

    initial: IngestState = {"file_path": str(dest)}
    result = ingest_graph.invoke(initial)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {"message": "Ingested successfully", "chunks": result.get("n_chunks", 0)}

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

    

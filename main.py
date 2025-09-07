from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas, crud
from database import engine, get_db
from app import router as lg_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CREATE
@app.post("/qa/", response_model=schemas.QAResponse)
def create_qa(qa: schemas.QACreate, db: Session = Depends(get_db)):
    return crud.create_qa(db=db, qa=qa)

# READ all
@app.get("/qa/", response_model=list[schemas.QAResponse])
def read_qas(db: Session = Depends(get_db)):
    return crud.get_qas(db=db)

# READ one
@app.get("/qa/{qa_id}", response_model=schemas.QAResponse)
def read_qa(qa_id: int, db: Session = Depends(get_db)):
    db_qa = crud.get_qa(db, qa_id)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return db_qa

# UPDATE
@app.put("/qa/{qa_id}", response_model=schemas.QAResponse)
def update_qa(qa_id: int, qa: schemas.QACreate, db: Session = Depends(get_db)):
    db_qa = crud.update_qa(db, qa_id, qa)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return db_qa

# DELETE
@app.delete("/qa/{qa_id}")
def delete_qa(qa_id: int, db: Session = Depends(get_db)):
    db_qa = crud.delete_qa(db, qa_id)
    if not db_qa:
        raise HTTPException(status_code=404, detail="QA not found")
    return {"message": "QA deleted successfully"}

app.include_router(lg_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
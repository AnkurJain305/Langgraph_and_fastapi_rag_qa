from sqlalchemy.orm import Session
import models as models, schemas.schemas as schemas

# CREATE
def create_qa(db: Session, qa: schemas.QACreate):
    db_qa = models.QA(question=qa.question, answer=qa.answer)
    db.add(db_qa)
    db.commit()
    db.refresh(db_qa)   
    return db_qa

# READ (all)
def get_qas(db: Session):
    return db.query(models.QA).all()

# READ (one by id)
def get_qa(db: Session, qa_id: int):
    return db.query(models.QA).filter(models.QA.id == qa_id).first()

# UPDATE
def update_qa(db: Session, qa_id: int, qa: schemas.QACreate):
    db_qa = db.query(models.QA).filter(models.QA.id == qa_id).first()
    if db_qa:
        db_qa.question = qa.question
        db_qa.answer = qa.answer
        db.commit()
        db.refresh(db_qa)
    return db_qa

# DELETE
def delete_qa(db: Session, qa_id: int):
    db_qa = db.query(models.QA).filter(models.QA.id == qa_id).first()
    if db_qa:
        db.delete(db_qa)
        db.commit()
    return db_qa

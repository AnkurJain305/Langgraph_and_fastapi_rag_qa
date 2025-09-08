from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    
class QABase(BaseModel):
    question: str
    answer: str

class QACreate(QABase):
    pass

class QAResponse(QABase):
    id: int

    class Config:
        orm_mode = True 

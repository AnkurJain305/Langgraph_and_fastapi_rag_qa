from sqlalchemy import Column, Integer, String
from database import Base

class QA(Base):
    __tablename__ = "qa"   

    id = Column(Integer, primary_key=True, index=True)  
    question = Column(String, nullable=False)           
    answer = Column(String, nullable=False)             

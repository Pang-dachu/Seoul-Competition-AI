from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# 요기 수정하기 
# json으로 받기 

class TEducations(BaseModel):
    id: int
    name: str
    status: str
    price: str
    capacity: int
    registerStart: str
    registerEnd: str
    educationStart: str
    educationEnd: str
    url: str
    hits: int
    # reviewsCount: dict[]


class TchatHistories(BaseModel):
    id: int 
    question: str
    answer: str
    feedback: bool
    createdAt: datetime


class TModelUpdateData(BaseModel):
    educations: TEducations
    chatHistories: TchatHistories


@router.post("/")
def predict(data: TModelUpdateData):
    
    # answer = 문장나오는함수(data.question)
    # return {"answer": answer}

    return {"answer": 0}

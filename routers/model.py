from enum import Enum
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class StatusEnum(str, Enum):
    sign_future = "수강신청예정"
    sign_now = "수강신청중"
    sign_finished = "마감"

class TEducation(BaseModel):
    id: int
    name: str
    status = StatusEnum
    price: int
    capacity: int
    registerStart: str
    registerEnd: str
    educationStart: str
    educationEnd: str
    url: str
    hits: int

class TChatHistory(BaseModel):
    id: int
    question: str
    answer: str
    feedback: Optional[bool] = None
    createdAt: datetime

class TModelUpdateData(BaseModel):
    educations: list[TEducation]
    chatHistories: list[TChatHistory]


@router.post("/")
def predict(data: TModelUpdateData):

    print("fastapi print - ", data)
    # answer = 문장나오는함수(data.question)
    # return {"answer": answer}

    return {"answer": 0}

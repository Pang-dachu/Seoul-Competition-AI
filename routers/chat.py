from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TChatQuestion(BaseModel):
    question: str

@router.post("/answer")
def predict(data: TChatQuestion):
    
    # answer = 문장나오는함수(data.question)
    # return {"answer": answer}

    return {"answer": 0}

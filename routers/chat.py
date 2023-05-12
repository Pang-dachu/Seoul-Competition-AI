from fastapi import APIRouter
from pydantic import BaseModel
from chat import chat

from recommend import model_update, user_recommend

router = APIRouter()

class TChatQuestion(BaseModel):
    question: str

@router.post("/answer")
def predict(data: TChatQuestion):
    
    answer = chat.use_chatbot(data.question)
    return {"answer": answer}
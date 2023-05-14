from fastapi import APIRouter
from pydantic import BaseModel
from chat import chat

router = APIRouter()


class TChatQuestion(BaseModel):
    question: str


@router.post("/answer")
def predict(data: TChatQuestion):

    answer = chat.use_chatbot(data.question)
    return {"answer": answer}

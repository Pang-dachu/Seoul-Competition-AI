from fastapi import APIRouter
from pydantic import BaseModel


class TextData(BaseModel):
    text: str

router = APIRouter()


@router.get("/my-route")
async def my_route():
    return {"message": "Hello, World!"}


def recommendation(text):
    return text

@router.post("/predict")
def predict(text_data: TextData):
    prediction = recommendation(text_data.text)
    return {"prediction": prediction}

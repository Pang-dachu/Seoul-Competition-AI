from fastapi import APIRouter
from pydantic import BaseModel


class TextData(BaseModel):
    text: str

router = APIRouter()

@router.get("/")
async def my_route():
    return {"admin": "is working"}


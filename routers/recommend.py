from fastapi import APIRouter
from pydantic import BaseModel
from recommend import model_update, user_recommend

router = APIRouter()

class TSearchKeyword(BaseModel):
    searchKeyword: str

class TOriginId(BaseModel):
    originId: int


@router.post("/searchKeyword")
def predict(data: TSearchKeyword):
    """
    - searchKeyword와 함께 POST 요청 받음
    """
    model_update.check_model_data()

    results = user_recommend.edu_recommend(data.searchKeyword)
    return {"results": results}

@router.post("/originId")
def predict(data: TOriginId):
    """
    - TOriginId 함께 POST 요청 받음
    """
    model_update.check_model_data()

    results = user_recommend.edu_recommend(data.originId)
    return {"results": results}


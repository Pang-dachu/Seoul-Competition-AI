from fastapi import APIRouter
from pydantic import BaseModel
from recommend import model_update, user_recommend

router = APIRouter()

class TSearchKeyword(BaseModel):
    searchKeyword: str

class TeducationId (BaseModel):
    educationId: int


@router.post("/searchKeyword")
def predict(data: TSearchKeyword):
    """
    - searchKeyword와 함께 POST 요청 받음
    """
    model_update.check_model_data()

    results = user_recommend.edu_recommend(data.searchKeyword)
    return {"results": results}

@router.post("/educationId")
def predict(data: TeducationId):
    """
    - TOriginId 함께 POST 요청 받음
    """
    model_update.check_model_data()

    results = user_recommend.id_edu_recommend(data.educationId)
    return {"results": results}


from fastapi import APIRouter
from pydantic import BaseModel
from recommend import model_update, user_recommend


class TSearchKeyword(BaseModel):
    searchKeyword: str

router = APIRouter()

'''
- 일주일에 한번 응답
- TfidfVectorizer 모델 /server 에 저장
- DateFrame /server 에 저장
'''
@router.get("/recreate")
def my_route():
    model_update.model_update()
    return {"message": "success"}

"""
- searchKeyword와 함께 POST 요청 받음
"""
@router.post("/predict")
def predict(data: TSearchKeyword):
    prediction = user_recommend.edu_recommend(data.searchKeyword)
    return {"prediction": prediction}

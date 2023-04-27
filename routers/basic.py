from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional


class TextData(BaseModel):
    text: str

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "nice"}



fake_items_db = [
  {
    'id': '31622628',
    'name': "[주거]초록쉼표, 식물과 함께하는 일상",
    'registerStart': "2023.05.26",
    'registerEnd': "2023.05.31",
    'educationStart': "2023.06.08",
    'educationEnd': "2023.06.22",
    'capacity': 20.0,
    'status': "수강신청예정",
    'price': 10000,
    'link': "https://50plus.or.kr/education-detail.do?id=31622628",
  },
  {
    'id': '31623735',
    'name': "[건강]몸신의 허리통증 없애는 비결",
    'registerStart': "2023.05.26",
    'registerEnd': "2023.05.31",
    'educationStart': "2023.06.13",
    'educationEnd': "2023.07.04",
    'capacity': 15.0,
    'status': "수강신청예정",
    'price': 10000,
    'link': "https://50plus.or.kr/education-detail.do?id=31623735",
  },
  {
    'id': '32130034',
    'name': "[커리어개발]유튜브 편집_파워디렉터 마스터",
    'registerStart': "2023.05.26",
    'registerEnd': "2023.06.04",
    'educationStart': "2023.06.10",
    'educationEnd': "2023.07.08",
    'capacity': 16.0,
    'status': "수강신청예정",
    'price': 20000,
    'link': "https://50plus.or.kr/education-detail.do?id=32130034",
  }]

# 데이터를 리스트로 뿌려주기
@router.get('/items')
async def list_items(skip: int = 0, limit: int = 2):
    return fake_items_db[skip:skip + limit]


# 옵션으로 query parameter를 분기 설정하기
@router.get('/items/{item_id}')
async def get_items(item_id: str, q: Optional[str] = None):
    if q:
        return {'item_id': item_id, 'q': q}
    return {'item_id': item_id}

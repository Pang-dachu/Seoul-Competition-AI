from fastapi import APIRouter
from pydantic import BaseModel
from recommend import model_update, user_recommend
import sched
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import logging
import datetime


router = APIRouter()
logger = logging.getLogger('logger')

class TSearchKeyword(BaseModel):
    searchKeyword: str

sched = BackgroundScheduler(timezone='Asia/Seoul')
# (trigger='cron', day='last', hour='4') 매월 말일 새벽 4시에 실행
@sched.scheduled_job(trigger='interval',  minutes=1)
def job():
    '''
    - 마지막 날에 실행
    - TfidfVectorizer 모델 /server 에 저장
    - DateFrame /server 에 저장
    - 한달치 데이터 분석 및 응답
    '''
    # 한달동안 추가된 educations 데이터를 받아 TF-IDF Vectorizer 파일 업데이트
    now = datetime.datetime.now()
    first_day = datetime.date(now.year, now.month, 1)
    formatted_date = first_day.strftime('%Y-%m-%d')
    url = f"http://spring:8080/api/v1/educations?createdAt={formatted_date}"
    model_update.model_update()

    # user_search table, user_view_detail table의 한달치 데이터를 받아서 데이터 분석
    url = "http://spring:8080/api/v1/아직미정"
    response = requests.get(url)
    data = response.json()

    return data

sched.start()


@router.post("/predict")
def predict(data: TSearchKeyword):
    """
    - searchKeyword와 함께 POST 요청 받음
    """

    prediction = user_recommend.edu_recommend(data.searchKeyword)
    return {"prediction": prediction}

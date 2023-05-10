#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
from joblib import dump
from konlpy.tag import Mecab
from datetime import datetime as dt

import numpy as np
import pandas as pd

import re
import requests
import json

import warnings
warnings.filterwarnings("ignore")

# 모델 및 df 존재 확인 
def check_model_data() :
    '''
    - 모델 및 data가 존재하는지 확인 
    - 존재하지 않는 경우 모델 및 data 생성 
    '''
    # 모델 저장 경로 : '/data/tfidf.pkl'
    # 데이터 저장 경로 : '/data/data.pkl'
    model = 'data/tfidf.pkl'
    data  = 'data/data.pkl'

    # 모델과 데이터 존재 확인 
    if os.path.isfile(model) and os.path.isfile(data):
        # print("파일 있음")
        return True
    else :
        #print("파일 없음")
        # 최초 모델 및 데이터 생성 코드 실행 추가 
        init_model_data()


# 최초 모델 및 df 생성 
def init_model_data() :
    '''
    - 전체 데이터에 대한 모델 생성 및 데이터 프레임 저장 
    - 교육 정보 backend에 전체 데이터 요청하는 부분 맞추기 
    
    
    return 없음 : model, dataframe 두 개의파일 pkl로 저장 
    '''
    # 교육 정보 backend에 현재 존재하는 전체 데이터 요청 코드 작성 
    
    # 요청 및 json 데이터 변환
#     response = requests.get(DATA_URL)
#     response_data = response.content.decode()
#     json_data = json.loads(response_data)
#     data = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
    pages = 0
    sprint_URL = f"http://spring:8080/api/v1/educations?page={pages}"
    response = requests.get(sprint_URL)
    response_data = response.content.decode()
    json_data = json.loads(response_data)

    pages = json_data["totalPages"] 

    data = pd.DataFrame()

    for page in range(0, pages+1) : 
        sprint_URL = f"http://spring:8080/api/v1/educations?page={page}"

        response = requests.get(sprint_URL)
        response_data = response.content.decode()
        json_data = json.loads(response_data)
        temp_data = pd.DataFrame(json_data['data'])

        data = pd.concat( [data, temp_data] )

    # 받아온 데이터에 대한 컬럼명, 전처리 등 수행 
    data = date_preprocessing(data)
    data = data_preprocessing(data)

    # 저장 
    save_model(data)
    save_dataframe(data)    


# 모델 업데이트 
def update_model_data(response) :
    # 스케쥴러를 통한 한달 단위 기준 모델 및 데이터 업데이트 
    # 1. 데이터 : 기존 데이터에 추가, 중복 확인
    # 2. 모델 : 생성된 데이터에 대하여 Vectorizor 모델 재 생성 후 저장 
    
    # 데이터 요청 
    # 요청은 아마 routers에서 받아오는 것으로 예상 
    
    # 기존 코드에서 사용되던 부분 -> 수정해서 사용하면 될 듯 
#     response = requests.get(DATA_URL)
    response_data = response.content.decode()
    json_data = json.loads(response_data)
    add_data = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
    
    # 추가된 데이터의 전처리 수행 
    # 1. 날짜형식 
    # 2. 불용어 처리 및 형태소 분리
    
    # 1. 날짜 형식 변경 
    add_data = date_preprocessing(add_data)
    
    # 2. 불용어 처리 및 형태소 분리 
    add_data = data_preprocessing(add_data)
    

    # 기존 데이터에 받아온 데이터 추가 
    # data : 기존 데이터 
    # add_data : 추가된 데이터 
    
    # 함수 수정할 것 
    
    # 기존 데이터 로드
    path = os.path.join(os.getcwd(), 'data', '/data/data.pkl')
    data = pd.read_pickle(path)
    
    data = pd.concat([data, add_data])
    
    # 중복 행 제거 
    data = data.drop_duplicates()
    
    # 모델 재생성, 데이터 재생성 -> 저장
    save_model(data)
    save_dataframe(data)

def date_preprocessing(dataframe) :
    '''
    - 데이터에 대한 날짜 형식 변경

    dataframe  : dataframe

    return : dataframe

    '''
    ## 날짜 정보 datetime

    # 표현 형식 변경
    dataframe["registerStart"] = dataframe["registerStart"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["registerEnd"] = dataframe["registerEnd"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["educationStart"] = dataframe["educationStart"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["educationEnd"] = dataframe["educationEnd"].apply(lambda x : re.sub(r"\.", r"-", x) )

    # datetime 형태 변경
    date_trans_col = ["registerStart","registerEnd","educationStart","educationEnd"]

    for col in date_trans_col :
        dataframe[col] = pd.to_datetime( dataframe[col] )

    return dataframe


# 불용어 처리
def clean_sentence(sentence) :
    '''
    - 데이터 분석 이후 문장에서 의미가 없을 것으로 판단되는 단어 불용어로 판단하여 처리

    sentence : Series

    return : Series

    '''

    # 날짜, 기수, 차수 제거
    sentence = re.sub(r"[0-9]+년", r" ", sentence)
    sentence = re.sub(r"[0-9]+차", r" ", sentence)
    sentence = re.sub(r"[0-9]+기", r" ", sentence)
    sentence = re.sub(r"[0-9]+월", r" ", sentence)
    sentence = re.sub(r"[0-9]+일", r" ", sentence)
    sentence = re.sub(r"[0-9]{1,2}.[0-9]{1,2}", r" ", sentence)

    # (주) , (요일)
    sentence = re.sub(r"\(+[가-힣]+\)", r" ", sentence)
    sentence = re.sub(r"[가-힣]째주", r" ", sentence)
    sentence = re.sub(r"[가-힣]{1}요일", r" ", sentence)

    # 마감 키워드 필요 없음
    sentence = re.sub(r"마감", r" ", sentence)

    # 50이라는 숫자 필요 없음
    sentence = re.sub(r"50", r" ", sentence)
    # 자격증 n급 필요 없을듯
    sentence = re.sub(r"[0-9]+급", r" ", sentence)
    # n단계도 필요 없을듯
    sentence = re.sub(r"[0-9]+단계", r" ", sentence)
    sentence = re.sub(r"[^0-9가-힣a-zA-Z]", r" ", sentence)

    return sentence


def tokenize(original_sent):
    '''
    - Mecab 형태소 분석기를 사용하여 문장를 "명사" 단위로 분류
    - 현 데이터는 문장의 의미보다는 사용되는 핵심 단어가 중요할 것으로 판단하여 결정

    sentence : Series

    return : Series

    '''

    tokenizer = Mecab()

    # tokenizer를 이용하여 original_sent를 토큰화하여 tokenized_sent에 저장하고, 이를 반환합니다.
    sentence = original_sent.replace('\n', '').strip()

    # tokenizer.nouns(sentence) -> 명사만 추출
    tokens = tokenizer.nouns(sentence)

    tokens = ' '.join(tokens)

    return tokens

def data_preprocessing(dataframe) :
    '''
    -  정의된 불용어 처리, 토크나이저를 데이터에 적용

    dataframe : dataframe

    return : dataframe

    '''
    # 교육명 불용어 처리하여 clean_sentence 컬럼으로 생성
    dataframe["clean_sentence"] = dataframe["name"].apply(lambda x : clean_sentence(x) )

    # 교육명 mecab 명사 토크나이징하여 mecab 컬럼으로 생성
    dataframe["mecab"] = dataframe["clean_sentence"].apply(lambda x : tokenize(x) )

    return dataframe

def save_model(data) :
    '''
    -  전체 데이터에 대한 tf-idf 모델 생성 후 저장

    data : dataframe

    '''
    path = os.path.join(os.getcwd(), 'data','tfidf.pkl')
    tfidf_vector = TfidfVectorizer().fit( data["mecab"] )
    dump(tfidf_vector, path)


def save_dataframe(data) :
    path = os.path.join(os.getcwd(), 'data', 'data.pkl')
    data.to_pickle(path)







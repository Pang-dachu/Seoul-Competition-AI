#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib
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
        

# 모델 업데이트 함수 
def get_dataframe(API_KEY : str , DATA_URL : str) :
    ''' 
    - 초기 모델 생성을 위하여 공공데이터를 json 형태로 받아와 데이터프레임으로 생성.
    - 이후 백엔드에서 데이터를 받아오는 과정으로 변경시 사용하지 않음.
    
    API_KEY : 공공 데이터의 API를 받아오기 위한 개인 키 값.
    DATA_URL : 공공 데이터의 API URL 값 
    
    return : pd.DataFrame 
    
    '''
    data = pd.DataFrame()
    
    # 요청 및 json 데이터 변환 
    response = requests.get(DATA_URL)
    response_data = response.content.decode()
    json_data = json.loads(response_data)
    
    # 데이터 row 갯수 확인
    # 데이터는 한번에 1000개 단위로만 요청이 가능함.
    list_total_count = json_data[list( json_data.keys() )[0]]['list_total_count']
    
    # 요청 갯수 제한에 따른 반복 실행하여 데이터 프레임 생성
    count = list_total_count // 1000
    les_count = list_total_count % 1000

    # 반복을 통한 데이터 프레임 생성 
    # 1000개 단위
    for i in range(count) :
        temp_url = DATA_URL[:-4]+str(1000*i + 1) + "/" + str(1000*(i+1)) 
        response = requests.get(temp_url)

        temp_data = response.content.decode()
        json_data = json.loads(temp_data)

        temp_df = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
        data = pd.concat( [data, temp_df] )
    
    # 1000개 단위로 반복 이후 나머지 갯수에 대한 추가 처리 
    temp_url =  DATA_URL[:-4]+str(1000*count + 1) + "/" + str(1000*count + les_count) 
    response = requests.get(temp_url)

    temp_data = response.content.decode()
    json_data = json.loads(temp_data)
    
    temp_df = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
    
    # 1000개 단위, 나머지 단위에 대한 데이터 병합
    data = pd.concat( [data, temp_df] )
    
    return data


def concat_data(data1, data2 ) : 
    '''
    - 초기 모델 생성을 위하여 공공데이터를 json 형태로 받아와 데이터프레임으로 생성.
    - 이후 백엔드에서 데이터를 받아오는 과정으로 변경시 사용하지 않음.
    - 초기에 사용하는 데이터가 2개이므로 병합하는 과정이 필요했음.
    
    data1 : DataFrame
    data2 : DataFrame
    
    return : pd.DataFrame 
    
    '''
    # 컬럼명 통일 시키는 과정
    data1.columns = ['교육넘버', '교육명', '교육신청시작일', '교육신청종료일', '교육시작일', '교육종료일', "수업시간", '수강정원', '교육상태', '교육비용', '강좌상세화면']
    data2.columns = ["교육넘버", "교육명", "교육시작일", "교육종료일", "교육신청시작일", "교육신청종료일", "수강정원", "교육비용", "교육상태", "강좌상세화면"]
    
    # 컬럼명 순서 통일 
    col_sort = ['교육넘버', '교육명', '교육신청시작일', '교육신청종료일', '교육시작일', '교육종료일',  '수강정원','교육상태', '교육비용', '강좌상세화면']
    
    data_1 = data1[ col_sort ]
    data_2 = data2[ col_sort ]
    # 이후 concat 진행 
    data = pd.concat([data_1, data_2])

    return data


def date_preprocessing(dataframe) :
    ''' 
    - 두 개의 데이터 프레임이 날짜 표현을 서로 다른 방식으로 표현함
    - 신청 가능한 교육을 날짜 기준으로 선정할 예정이므로 datetime을 사용하기 위해 날짜 형식 변경
    - 이후 백엔드에서 데이터를 받아오는 경우 수정되거나 사용되지 않을 수 있음.
    
    dataframe  : dataframe
    
    return : dataframe
    
    '''
    ## 날짜 정보 datetime

    # 표현 형식 변경
    dataframe["교육신청시작일"] = dataframe["교육신청시작일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육신청종료일"] = dataframe["교육신청종료일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육시작일"] = dataframe["교육시작일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육종료일"] = dataframe["교육종료일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    
    # int, datetime 형태 변경 
    date_trans_col = ["교육신청시작일","교육신청종료일","교육시작일","교육종료일"]

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
    dataframe["clean_sentence"] = dataframe["교육명"].apply(lambda x : clean_sentence(x) )
    
    # 교육명 mecab 명사 토크나이징하여 mecab 컬럼으로 생성
    dataframe["mecab"] = dataframe["clean_sentence"].apply(lambda x : tokenize(x) ) 

    return dataframe

def save_model(data) :
    '''
    -  전체 데이터에 대한 tf-idf 모델 생성 후 저장 
    
    data : dataframe
    
    '''
    tfidf_vector = TfidfVectorizer().fit( data["mecab"] )
    dump(tfidf_vector, 'tfidf.pkl')
    

def save_dataframe(data) :
    
    data.to_pickle('data.pkl')


def model_update() :
    '''
    - 모델 업데이트 
    - 일정 주기로 업데이트 시 실행할 수 있도록 
    - .py 파일을 따로 생성하는 방법도 고려 
    
    input_data : str
    data : dataframe
    vectorizer : TfidfVectorizer
    
    return : str

    '''   

    API_KEY = "61484f6245666f7838344a79694e77"

    서울시50플러스포털교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/FiftyPotalEduInfo/1/5/"
    서울시어르신취업지원센터교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/tbViewProgram/1/5/"
    
    data_01 = get_dataframe(API_KEY, 서울시50플러스포털교육정보)
    data_02 = get_dataframe(API_KEY, 서울시어르신취업지원센터교육정보)
    total_data = concat_data(data_01, data_02)
    
    total_data = date_preprocessing(total_data)
    total_data = data_preprocessing(total_data)
    
    save_model(total_data)
    save_dataframe(total_data)
    
    print("모델 업데이트 완료")


## 실행

model_update()






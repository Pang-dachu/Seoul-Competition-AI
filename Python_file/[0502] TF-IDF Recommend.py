#!/usr/bin/env python
# coding: utf-8

# ### Import PART

# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from konlpy.tag import Mecab
from datetime import datetime as dt

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf 
import re
import requests
import json

import warnings
warnings.filterwarnings("ignore")


# ### json_to_data PART

# In[3]:


## 필요 라이브러리 

API_KEY = "61484f6245666f7838344a79694e77"

서울시50플러스포털교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/FiftyPotalEduInfo/1/5/"
서울시어르신취업지원센터교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/tbViewProgram/1/5/"

import requests
import json

API_KEY = "61484f6245666f7838344a79694e77"


# 두개의 데이터에 대해서 받을수 있는 URL 주소이므로 
# 다른 URL 사용시에 URL 끝 부분에 대한 확인 필요
서울시50플러스포털교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/FiftyPotalEduInfo/1/5/"
서울시어르신취업지원센터교육정보 = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/tbViewProgram/1/5/"

def get_dataframe(API_KEY, DATA_URL) :
    url = DATA_URL
    data = pd.DataFrame()
    
    response = requests.get(url)
    response_data = response.content.decode()
    json_data = json.loads(response_data)
    
    # 데이터 row 갯수 확인
    # 한번에 1000개 단위로 받아오는 것이 가능
    list_total_count = json_data[list( json_data.keys() )[0]]['list_total_count']
    
    # 데이터 갯수에 따라 반복 실행하여 DataFrame 생성하기 
    count = list_total_count // 1000
    les_count = list_total_count % 1000

    # 데이터 갯수에 따른 요청 및 데이터 프레임 생성 과정
    for i in range(count) :
        temp_url = url[:-4]+str(1000*i + 1) + "/" + str(1000*(i+1)) 
        response = requests.get(temp_url)

        temp_data = response.content.decode()
        json_data = json.loads(temp_data)

        temp_df = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
        data = pd.concat( [data, temp_df] )
    
    temp_url =  url[:-4]+str(1000*count + 1) + "/" + str(1000*count + les_count) 
    response = requests.get(temp_url)

    temp_data = response.content.decode()
    json_data = json.loads(temp_data)

    temp_df = pd.json_normalize(json_data[list( json_data.keys() )[0]]['row'])
    data = pd.concat( [data, temp_df] )
    
    return data


# In[ ]:


# 두가지 데이터에 대한 결합 
def concat_data(data1, data2) : 
    # 컬럼명 통일 시키는 과정 필요 
    data1.columns = ['교육넘버', '교육명', '교육신청시작일', '교육신청종료일', '교육시작일', '교육종료일', "수업시간", '수강정원', '교육상태', '교육비용', '강좌상세화면']
    data2.columns = ["교육넘버", "교육명", "교육시작일", "교육종료일", "교육신청시작일", "교육신청종료일", "수강정원", "교육비용", "교육상태", "강좌상세화면"]
    
    # 컬럼명 순서 통일 
    col_sort = ['교육넘버', '교육명', '교육신청시작일', '교육신청종료일', '교육시작일', '교육종료일',  '수강정원','교육상태', '교육비용', '강좌상세화면']
    
    data_1 = data1[ col_sort ]
    data_2 = data2[ col_sort ]
    # 이후 concat 진행 
    data = pd.concat([data_1, data_2])
    
    # data return 
    return data

data_01 = get_dataframe(API_KEY, 서울시50플러스포털교육정보)
data_02 = get_dataframe(API_KEY, 서울시어르신취업지원센터교육정보)

data_01.shape, data_02.shape

total_data = concat_data(data_01, data_02)


# ### preprocessing PART

# In[4]:


# 불용어 처리 
def clean_sentence(sentence) :
    # 날짜, 기수, 차수 제거 
    sentence = re.sub(r"[0-9]+년", r" ", sentence)
    sentence = re.sub(r"[0-9]+차", r" ", sentence)
    sentence = re.sub(r"[0-9]+기", r" ", sentence)
    sentence = re.sub(r"[0-9]+월", r" ", sentence)
    sentence = re.sub(r"[0-9]+일", r" ", sentence)
    sentence = re.sub(r"[0-9]{1,2}.[0-9]{1,2}", r" ", sentence)
    
    # (주) , (요일)
    sentence = re.sub(r"\(+[가-힣]+\)", r" ", sentence)
    #sentence = re.sub(r"[/s]\(.\)[/s]", r" ", sentence)
    
    # 주차, 요일 형식 제거 
    # sentence = re.sub(r"[가-힣]{2}주", r" ", sentence) 
    # "알려주는" 단어에 영향이 생김
    
    sentence = re.sub(r"[가-힣]째주", r" ", sentence) 
#     sentence = re.sub(r"둘째주", r" ", sentence) 
#     sentence = re.sub(r"셋째주", r" ", sentence) 
#     sentence = re.sub(r"넷째주", r" ", sentence) 
    
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


# In[5]:


def tokenize(original_sent, nouns=False):
    tokenizer = Mecab()

    # tokenizer를 이용하여 original_sent를 토큰화하여 tokenized_sent에 저장하고, 이를 반환합니다.
    sentence = original_sent.replace('\n', '').strip()
    if nouns:       
        # tokenizer.nouns(sentence) -> 명사만 추출
        tokens = tokenizer.nouns(sentence)
    else:
        tokens = tokenizer.morphs(sentence)
        
    tokens = ' '.join(tokens)
    
    return tokens


# In[6]:


def date_preprocessing(dataframe) :

    ## 날짜 정보 datetime

    # 표현 형식 변경
    dataframe["교육신청시작일"] = dataframe["교육신청시작일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육신청종료일"] = dataframe["교육신청종료일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육시작일"] = dataframe["교육시작일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    dataframe["교육종료일"] = dataframe["교육종료일"].apply(lambda x : re.sub(r"\.", r"-", x) )
    
    # int, datetime 형태 변경 
    #dataframe = dataframe.astype({"수강정원" : "int"})
    date_trans_col = ["교육신청시작일","교육신청종료일","교육시작일","교육종료일"]

    for col in date_trans_col : 
        dataframe[col] = pd.to_datetime( dataframe[col] )
        
    # 교육명 불용어 처리 
    dataframe["clean_sentence"] = dataframe["교육명"].apply(lambda x : clean_sentence(x) )
    
    # 교육명 mecab 명사 토크나이징
    dataframe["mecab"] = dataframe["clean_sentence"].apply(lambda x : tokenize(x, True) ) 
    
    # TF-IDF 방식을 사용하기 위해 mecab 컬럼에 대해 각 데이터 string 형태로 변경 
    # dataframe["mecab"] = dataframe["mecab"].apply(lambda x : str(x) for x in dataframe["mecab"])
    
    return dataframe


# ### Similarity PART

# In[7]:


# 코사인 유사도 
def l1_normalize(v):
  norm = np.sum(v)
  return v / norm


# In[8]:


def cosine_similarity_value(vec_1, vec_2):
  return round(cosine_similarity(vec_1, vec_2)[0][0], 3)


# In[9]:


def possible_edu (dataframe) :
    today = f"{dt.today().year}-{dt.today().month}-{dt.today().day}"
    
    # 수강 신청이 가능한 경우 
    # 1. 교육 상태가 마감이 아닌 경우 
    cond_01 = (dataframe["교육상태"] == "마감")

    # 2. 교육 신청 종료일이 현재 날짜를 지나지 않은 경우
    cond_02 = (dataframe["교육신청종료일"] > today)
    
    temp_data = dataframe.loc[ ~cond_01 & cond_02 ]

    return temp_data


# In[20]:


def edu_recommend(input_data, data, vectorizer) :
    
    # 입력 단어에 대한 임시 데이터 프레임 생성    
    temp = pd.DataFrame({
        # "교육넘버" : "0000",
        "교육명": [input_data],
        "clean_sentence" : clean_sentence(input_data),
         "mecab" : ["123"]
    })

    temp["mecab"] = temp["clean_sentence"].apply(lambda x : tokenize(x, True) )
    # temp["mecab"] = temp["mecab"].apply(lambda x : str(x) for x in temp["mecab"])
    
    # 검색 단어를 포함한 전체 데이터 프레임 
    temp_total_data = data[::]
    
    temp_total_data = pd.concat([temp_total_data,temp])
    temp_total_data = temp_total_data.reset_index( drop=True )
    
    # TF-IDF 벡터화 
    #tfidf_vectorizer = TfidfVectorizer()
    #tfidf_mecab = tfidf_vectorizer.fit( temp_total_data["mecab"] )
    tfidf_vector = vectorizer.transform( temp_total_data["mecab"] )
    tfidf_norm_l1 = l1_normalize(tfidf_vector)
    
    
    # 검색 단어 
    target = tfidf_norm_l1[-1]
    
    # 코사인 유사도 적용
    cosin_result = []

    for i in tfidf_norm_l1 :
        cosin_result.append( cosine_similarity_value(target, i) )
        
    temp_total_data["cosin"] = cosin_result


    temp = temp_total_data.loc[ temp_total_data["cosin"] > 0 ]
    temp = temp.sort_values(["cosin"], ascending=False)[1:6]
    
    if temp.empty :
            print("추천 정보가 없습니다.")
            exit()

    for i,j in zip(temp["교육명"], temp["cosin"]):
        print( i, j )


# ### Excute PART

# In[13]:


total_data = date_preprocessing(total_data)


# In[27]:


total_data.shape


# In[14]:


total_data.head()


# In[17]:


today_edu = possible_edu( total_data )


# In[26]:


today_edu.shape


# In[15]:


# 전체 데이터에 대한 TF-IDF Vectorizer
# vectorizer = TfidfVectorizer()

tfidf_vector = TfidfVectorizer().fit( total_data["mecab"] )


# ### Example PART

# In[21]:


x = "일자리"

edu_recommend(x, today_edu, tfidf_vector)


# In[22]:


x = "경비원"

edu_recommend(x, today_edu, tfidf_vector)


# In[23]:


x = "코딩"

edu_recommend(x, today_edu, tfidf_vector)


# In[24]:


x = "관리"

edu_recommend(x, today_edu, tfidf_vector)


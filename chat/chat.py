# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import os
import joblib

def check_chat_data() :
    '''
    - 챗봇 모델 및 data가 존재하는지 확인 
    - 존재하지 않는 경우 챗봇 사용 제한 ?

    return True/False
    '''
    # 챗봇 모델 저장 경로 : '/data/chatbot_model.pkl'
    # 데이터 저장 경로 : '/data/chatbot_data.pkl'
    model = 'data/chatbot_model.pkl'
    data  = 'data/chatbot_data.pkl'

    # 모델과 데이터 존재 확인 
    if os.path.isfile(model) and os.path.isfile(data):
        # print("파일 있음")
        return True
    else :
        #print("파일 없음")
        # 최초 모델 및 데이터 생성 코드 실행 추가 
        return False

def load_chatbot() :
    '''
    - 챗봇 모델 로드 

    return : 챗봇 모델 
    '''
    path = os.path.join(os.getcwd(), 'data', 'chatbot_model.pkl')
    chatbot_model = joblib.load(path)

    return chatbot_model


def load_chatbot_data() :
    '''
    - 챗봇 데이터 로드  

    return : 챗봇 데이터 
    '''
    path = os.path.join(os.getcwd(), 'data', 'chatbot_data.pkl')
    chatbot_data = pd.read_pickle(path)

    return chatbot_data


def use_chatbot(user_question) :
    '''
    - 챗봇 모델 및 데이터 존재하는지 확인 : 미존재시 챗봇 이용 불가 상태 반환
    - 챗봇 모델에서 적합한 데이터 (str) 반환

    user_question : str

    return : 챗봇 모델 
    '''
    if check_chat_data()==True :
        chatbot_model = load_chatbot()
        chatbot_data  = load_chatbot_data()
        answer = 1
        # input_embed = chatbot_model.encode(user_question)
        # chatbot_data["cosin"] = chatbot_data["embedding"].map(lambda x : cosine_similarity([input_embed],[x]).squeeze())

        # if chatbot_data["cosin"].max() < 0.60 :
        #     answer = "말씀하신 내용을 정확히 이해하지 못했어요.\n다른 문장으로 물어봐주시겠어요?"
        #     return answer 

        # temp = chatbot_data.sort_values(["cosin"], ascending=False)[0:1]
        # answer = ''.join([x for x in temp["answer"]])

    else :
        answer = "챗봇이 잠시 쉬러갔어요. 금방 돌아올게요!"
    
    return answer

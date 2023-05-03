# BASE IMAGE
FROM python:3.9

WORKDIR /server

COPY ./requirements.txt /server/requirements.txt

# konlpy, py-hanspell, soynlp 패키지 설치
RUN pip install konlpy

# 형태소 분석기 mecab 설치
RUN cd /tmp && \
    wget "https://www.dropbox.com/s/9xls0tgtf3edgns/mecab-0.996-ko-0.9.2.tar.gz?dl=1" && \
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz?dl=1 && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make check && \
    make install && \
    ldconfig

RUN cd /tmp && \
    wget "https://www.dropbox.com/s/i8girnk5p80076c/mecab-ko-dic-2.1.1-20180720.tar.gz?dl=1" && \
    apt install -y autoconf && \
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz?dl=1 && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig

# 형태소 분석기 mecab 파이썬 패키지 설치
RUN cd /tmp && \
    git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git && \
    cd mecab-python-0.996 && \
    python setup.py build && \
    python setup.py install

# 패키지
RUN pip install numpy pandas scikit-learn requests joblib

RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

COPY ./ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
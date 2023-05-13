# BASE IMAGE
FROM python:3.10-slim

WORKDIR /server

# jvm 설치
RUN apt-get update && apt-get install -y default-jre

# requirements.txt 설치
COPY ./requirements.txt /server/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

COPY ./ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
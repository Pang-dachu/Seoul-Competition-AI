# BASE IMAGE
FROM python:3.9

RUN mkdir /server
WORKDIR /server

COPY ./requirements.txt /server/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

COPY ./ ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
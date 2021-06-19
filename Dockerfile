FROM python:3.9.5-slim

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app ./

CMD ["sh", "-c", "streamlit run --server.port $PORT /usr/app/src/main.py"]

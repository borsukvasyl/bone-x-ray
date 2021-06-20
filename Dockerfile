FROM python:3.6-slim-stretch

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

CMD streamlit run --server.port $PORT app.py

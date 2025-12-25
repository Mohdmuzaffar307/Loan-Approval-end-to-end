FROM python:3.10-alpine

WORKDIR /application

COPY app/ /application/
COPY reports/ /application/reports/

RUN pip install -r requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run" ,"app.py" ]
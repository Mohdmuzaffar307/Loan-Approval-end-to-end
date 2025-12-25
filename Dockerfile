FROM python:3.10-slim

WORKDIR /application

COPY app/ /application/
COPY reports/ /application/reports/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run" ,"app.py" ]
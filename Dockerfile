FROM python:3.8

WORKDIR /app

# copy requirements.txt
COPY requirements.txt $MICRO_SERVICE/requirements.txt
# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# copy project
COPY . .
CMD ["sh", "-c", "streamlit run --server.port $PORT app.py"] 

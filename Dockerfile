FROM python:3.8

ENV MICRO_SERVICE=/home/app/webapp
# set work directory
RUN mkdir -p $MICRO_SERVICE
# where your code lives
WORKDIR $MICRO_SERVICE

# install dependencies
RUN pip install --upgrade pip
# copy project
COPY . $MICRO_SERVICE
RUN pip install -r requirements.txt
CMD ["sh", "-c", "streamlit run --server.port $PORT app.py"] 

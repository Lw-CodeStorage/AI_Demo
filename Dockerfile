
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#變數
ENV PORT 8080 
#Container 對外開放的port
EXPOSE 8080

# install dependencies
RUN pip install --upgrade pip 
COPY ./requirements.txt /usr/src/app
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app

CMD python manage.py runserver 0.0.0.0:$PORT 
#CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 AI_PlayGroud.wsgi:application
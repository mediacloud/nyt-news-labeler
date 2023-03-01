FROM python:3.7

MAINTAINER Rahul Bhargava <r.bhargava@northeastern.edu>

COPY . /nyt-news-labeler/

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y brotli
RUN apt-get install -y curl

WORKDIR /nyt-news-labeler/

RUN python3 -m pip install -r requirements.txt
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt
RUN python3 download_models.py

EXPOSE 8000

CMD [ "/usr/local/bin/gunicorn", "-b", ":8000", "-t", "900", "app:app" ]

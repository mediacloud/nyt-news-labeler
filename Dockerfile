FROM ubuntu:20.04

MAINTAINER Rahul Bhargava <r.bhargava@northeastern.edu>

ARG DEBIAN_FRONTEND=noninteractive

COPY . /nyt-news-labeler/

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y brotli

WORKDIR /nyt-news-labeler/

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt
RUN python3 download_models.py

EXPOSE 8000

CMD [ "/usr/local/bin/gunicorn", "-b", ":8000", "-t", "900", "app:app" ]

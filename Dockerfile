FROM ubuntu:20.04

MAINTAINER Rahul Bhargava <r.bhargava@northeastern.edu>

ARG DEBIAN_FRONTEND=noninteractive

COPY . /nyt-news-labeler/

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y brotli
RUN apt-get install -y curl

RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.7
WORKDIR /nyt-news-labeler/

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.7
RUN python3.7 -m pip install -r requirements.txt
RUN python3.7 -m nltk.downloader -d /usr/local/share/nltk_data punkt
RUN python3.7 download_models.py

EXPOSE 8000

CMD [ "/usr/local/bin/gunicorn", "-b", ":8000", "-t", "900", "app:app" ]

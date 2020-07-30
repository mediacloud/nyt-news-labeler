#
# Build image:
#
#     docker build -t nyt-news-labeler .
#
# Run image:
#
#     docker run -p 8000:8000 nyt-news-labeler
#
# Push image to Docker Hub:
#
#     docker tag nyt-news-labeler:latest dockermediacloud/nyt-news-labeler:<YYYYMMDD>
#     docker push dockermediacloud/nyt-news-labeler:<YYYYMMDD>
#

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

COPY . /nyt-news-labeler/

RUN \
    apt-get -y update && \
    apt-get -y --no-install-recommends install apt-utils && \
    apt-get -y --no-install-recommends install apt-transport-https && \
    apt-get -y --no-install-recommends install acl && \
    apt-get -y --no-install-recommends install sudo && \
    apt-get -y --no-install-recommends install brotli build-essential curl file python python-dev python-h5py python-pip python-setuptools && \
    apt-get -y clean && \
    \
    pip install --upgrade pip && \
    rm -rf /root/.cache/ && \
    \
    useradd -ms /bin/bash nytlabels && \
    echo 'nytlabels ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/99_nytlabels && \
    \
    mv /nyt-news-labeler /home/nytlabels/nyt-news-labeler && \
    chown -R nytlabels:nytlabels /home/nytlabels/

WORKDIR /home/nytlabels/nyt-news-labeler/

RUN \
    pip install -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt && \
    rm -rf /home/nytlabels/.cache/

USER nytlabels

RUN python download_models.py

EXPOSE 8000

ENTRYPOINT [ "/usr/local/bin/gunicorn", "-b", ":8000", "-t", "900", "app:app" ]

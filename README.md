NYT-Based News Tagger
=====================

A labeller for news articles trained on the [NYT annotated corpus](https://catalog.ldc.upenn.edu/ldc2008t19)
by Jasmin Rubinovitz as part of the [MIT Media Lab SuperGlue project](https://www.media.mit.edu/projects/superglue/overview/).
Give it the clean text of a story (i.e. no html content), and it returns various descriptors
and taxonomic classifiers based on models trained on the tagging in the NYT corpus.

Note - we have *not* formally assessed these models for embedded bias. Surely they have many, because they are based on
the Google News word2vec model and New York Times historical tagging. Be aware as you use results that they likely 
reflect historical American cultural biases in news reporting.

We use it in the [Media Cloud](https://mediacloud.org) project to automatically tag all news stories with the
themes we think they are about.

Running Via DockerHub
---------------------

The quickest path to running this is to fetch the latest release from DockerHub:

```
docker pull rahulbot/nyt-news-labeler:latest
docker run -p 8000:8000 -m 8G -d rahulbot/nyt-news-labeler:latest
```

Then just hit a `http://localhost:8080/` to test it out.


Local Dev Installation
----------------------

1. Install Python 3.x (we use pyenv: `pyenv install 3.8.2`)
2. Install python requirements: `pip install -r requirements.txt`
3. Install [brotli](https://brotli.org/index.html): `brew install brotli` (on MacOS) 
4. Download the models: `download_models.py` (this will take 10+ minutes, depending on your internet speed)


Run `./run.sh`. Note: this consumes about **8 GB of memory** while running, to keep all the models loaded up.

### Web Test Harness

This exposes a simple web UI to make testing easier. Visit `localhost:8000/` to try it out. You can paste any
raw text in, and click "Get Labels". In a second you will see the top 30 labels from each model below the input.

### API

For batch processing this exposes a simple API. You can make a request like this:
```
curl -X POST http://localhost:8000/predict.json -H "Content-Type: application/json" -d '{"text": "Federal agents show stronger force at Portland protests despite order to withdraw" }'
```
You will get back results like this:

```json
{
   "milliseconds":77.39500000000001,
   "predictions":{
      "allDescriptors":[
         {
            "label":"demonstrations and riots",
            "score":"0.28221"
         },
         {
            "label":"politics and government",
            "score":"0.03751"
         },
         ...
      ],
      "descriptors3000":[
         {
            "label":"company reports",
            "score":"0.74512"
         },
         {
            "label":"demonstrations and riots",
            "score":"0.64673"
         },
         ...
      ],
      "descriptors600":[
         {
            "label":"demonstrations and riots",
            "score":"0.65299"
         },
         {
            "label":"politics and government",
            "score":"0.09620"
         },
         ...
      ],
      "descriptorsAndTaxonomies":[
         {
            "label":"demonstrations and riots",
            "score":"0.43143"
         },
         {
            "label":"top/news",
            "score":"0.27492"
         },
         ...
      ],
      "taxonomies":[
         {
            "label":"Top/Features/Travel/Guides/Destinations/North America/United States/Oregon",
            "score":"0.35107"
         },
         {
            "label":"Top/News",
            "score":"0.18331"
         },
         ...
      ]
   },
   "status":"OK",
   "version":"1.1.0"
}
```


Releasing to Docker Hub
-----------------------

When you creating a new release, be sure to increment the `VERSION` constant in `app.py`. Then tag the repo with the 
same number. 

I build and release this to DockerHub for easier deployment on your server. To release the latest code I run:
```
docker build -t rahulbot/nyt-news-labeler .
docker push rahulbot/nyt-news-labeler
```

To release a tagged version, I something like this run:
```
docker build -t rahulbot/nyt-news-labeler:1.1.0 .
docker push rahulbot/nyt-news-labeler:1.1.0
```

To run a container I've built locally I do: 
```
docker run -p 8000:8000 -m 8G rahulbot/nyt-news-labeler
```

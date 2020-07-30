NYT-Based News Tagger
=====================

A labeller for news articles trained on the [NYT annotated corpus](https://catalog.ldc.upenn.edu/ldc2008t19)
by Jasmin Rubinovitz as part of the [MIT Media Lab SuperGlue project](https://www.media.mit.edu/projects/superglue/overview/).
Give it the clean text of a story (i.e. no html content), and it returns various descriptors
and taxonomic classifiers based on models trained on the tagging in the NYT corpus.

We use it in the [Media Cloud](https://mediacloud.org) project to automatically tag all news stories with the
themes we think they are about.


Installation
------------

1. Install Python 3.7.3 (we use pyenv: `pyenv install 3.7.3`)
2. Install python requirements: `pip install -r requirements.txt`
3. Download the models: `download_models.py` (this will take 10+ minutes, depending on your internet speed)


Usage
-----

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


Releasing
---------

When you creating a new release, be sure to increment the `VERSION` constant in `app.py`. Then tag the repo with the 
same number. 


Deploying
---------

This is built to deploy in a container (we use Dokku).  Set the `WORKERS` environment variable to set how many
workers gunicorn starts with. If you want to scale horizontally remember that each instance will want to load its own
copy of the models, so you'll need around 8 GB for each instance. 

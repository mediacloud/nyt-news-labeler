NYT-Based News Tagger
=====================

A labeller for news articles trained on the [NYT annotated corpus](https://catalog.ldc.upenn.edu/ldc2008t19)
by Jasmin Rubinovitz as part of the [MIT Media Lab SuperGlue project](https://www.media.mit.edu/projects/superglue/overview/).
Give it the clean text of a story (i.e. no html content), and it returns various descriptors
and taxonomic classifiers based on models trained on the tagging in the NYT corpus.

We use it in the Media Cloud project to automatically tag all news stories with the themes we think they are about.

Installation
------------

This is built with Python. 

On OSX I had to install brotli and hdf5 first with brew: `brew install brotli hdf5`.

Do this to install all the Python dependencies.

```shell
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
```

You also need the `word2vec` pre-trained Google News corpus and NYTLabels model.  Run `download_models.py` to get them.

Lastly, you'll need `punkt` dataset from NLTK data:

```shell
python -m nltk.downloader -d /usr/local/share/nltk_data punkt
```

Usage
-----

Simply do `run.sh`, or `gunicorn app:app -t 900` and then visit `localhost:8000/` to try it out.

Note: this consumes about **4 GB of memory** while running, to keep all the models loaded up.


Releasing
---------

When you creating a new release, be sure to increment the `VERSION` constant in `app.py`. Then tag the repo with the
same number. 


Deploying
---------

This is built to deploy in a container (we use Dokku).  Set the `WORKERS` environment variable to set how many
workers gunicorn starts with.

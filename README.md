News Tagger
===========

A labeller for news articles trained on the [NYT annotated corpus](https://catalog.ldc.upenn.edu/ldc2008t19)
by Jasmin Rubinowitz as part of the [MIT Media Lab SuperGlue project](https://www.media.mit.edu/projects/superglue/overview/).
Give it the clean text of a story (ie. no html content), and it returns various descriptors
and taxonomic classifiers based on models trained on the taging in the NYT corpus.

We use it in the Media Cloud project to automatically tag news stories with the themes we think they are about.

Installation
------------

This is built with Python. 

On OSX I had to install hdf5 first with brew: `brew install hdf5`.

Do this to install all the Python dependencies.

```shell
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt  
```

You also need the `word2vec` pre-trained Google News corpus.  Download 
[the `.bin` file](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and stick it in a
`word2vec-GoogleNews-vectors` folder.

Usage
-----

`gunicorn app:app -t 900`

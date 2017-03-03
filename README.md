# predict-news-labels

Trained multi label classifier to predict descriptors and taxonomies. Trained on the NYT Annotated corpus.


To run locally:
```bazaar

virtualenv venv
source venv/bin/activate
 
pip install -r requirments.txt  
  
gunicorn app:app -t 900


```

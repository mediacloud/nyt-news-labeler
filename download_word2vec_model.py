import os
import requests
import shutil

MODEL_GOOGLE_NEWS_URL = "https://s3.amazonaws.com/mediacloud-nytlabels-data/predict-news-labels/GoogleNews-vectors-negative300.bin"

model_dir = "./word2vec-GoogleNews-vectors"
model_name = "GoogleNews-vectors-negative300.bin"

path_to_model_file = os.path.join(model_dir, model_name)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)


# https://stackoverflow.com/a/39217788/1172063
def download_file(url, destination_file):
    r = requests.get(url, stream=True)
    with open(destination_file, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

if not os.path.isfile(path_to_model_file):
    print "Google word2vec model not found, downloading model file from the cloud..."
    download_file(MODEL_GOOGLE_NEWS_URL, path_to_model_file)
    print "  done!"
else:
    print "Google word2vec model already exists."

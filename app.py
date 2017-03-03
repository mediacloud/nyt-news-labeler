from flask import Flask, request, jsonify, render_template
import zipfile

import nltk

import numpy as np
from nltk.tokenize import word_tokenize
import os
import gensim
import urllib
from keras.models import load_model
import cPickle as pickle
import json

THRESHOLD = 0.05
PUNCTUATION = '.,:;!?()/\"-<>[]{}|\\@#`$%^&*'

def load_scaler_from_disk(path_to_disk):
    """ Load a pickle from disk to memory """
    if not os.path.exists(path_to_disk):
        raise ValueError("File " + path_to_disk + " does not exist")

    return pickle.load(open(path_to_disk, 'rb'))


class TopicDetectionModel():
    """
    Extract story topics, using a multilabel classifier trained on the New York Times annotated corpus
    """

    def __init__(self, keras_model, word2vec_model, scaler, labels):
        self.word2vec_model = word2vec_model
        self.keras_model = keras_model
        self.scaler = scaler
        self.labels = labels

    def predict(self, text):
        if type(self.keras_model.input) == list:
            _, sample_length, embedding_size = self.keras_model.input_shape[0]
        else:
            _, sample_length, embedding_size = self.keras_model.input_shape

        words = [w.lower() for w in word_tokenize(text)
                if w not in PUNCTUATION][:sample_length]
        x_matrix = np.zeros((1, sample_length, embedding_size))

        for i, w in enumerate(words):
            if w in self.word2vec_model:
                word_vector = self.word2vec_model[w].reshape(1, -1)
                scaled_vector = self.scaler.transform(word_vector, copy=True)[0]
                x_matrix[0][i] = scaled_vector

        if type(self.keras_model.input) == list:
            x = [x_matrix] * len(self.keras_model.input)
        else:
            x = [x_matrix]

        y_predicted = self.keras_model.predict(x)

        zipped = zip(self.labels, y_predicted[0])

        return sorted(zipped, key=lambda elem: elem[1], reverse=True)

# download and load Google News word2vec model
emmbedings_dir = "./word2vec-GoogleNews-vectors"
emmbedings_file_name = "GoogleNews-vectors-negative300.bin"
emmbedings_file_path = os.path.join(emmbedings_dir, emmbedings_file_name)

if not os.path.exists(emmbedings_dir):
    os.mkdir(emmbedings_dir)
if not os.path.isfile(emmbedings_file_path):
    print "Google word2vec model not found, downloading file..."
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/466924777/GoogleNews-vectors-negative300.bin", emmbedings_file_path)
print "Loading pre-trained word to vec model..."
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format(emmbedings_file_path, binary=True)
print "weord2vec Model loaded"

# download pre-trained models
models_dir = "./models"
models_file_name = "saved_models"
models_file_path = os.path.join(models_dir, models_file_name)

if not os.path.isdir(os.path.join(models_file_path)):
    print "Trained models not found, downloading files..."
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/466924777/nyt_labels/saved_models.zip", models_file_path+".zip")
    with zipfile.ZipFile(models_file_path+".zip","r") as zip_ref:
        zip_ref.extractall(models_dir)

# load pre-trained scaler
scaler = load_scaler_from_disk('./scaler/scaler')


descriptors_3000 = []
with open('./models/descriptors_3000.json') as data_file:
    descriptors_3000 = json.load(data_file)
keras_model3000 = load_model('./models/saved_models/descriptors_3000.hdf5')
model3000 = TopicDetectionModel(keras_model=keras_model3000, word2vec_model=word2vecmodel, scaler=scaler, labels=descriptors_3000)

descriptors_600 = []
with open('./models/descriptors_600.json') as data_file:
  descriptors_600 = json.load(data_file)
keras_model600 = load_model('./models/saved_models/descriptors_600.hdf5')
model600 = TopicDetectionModel(keras_model=keras_model600, word2vec_model=word2vecmodel, scaler=scaler, labels=descriptors_600)

all_descriptors = []
with open('./models/all_descriptors.json') as data_file:
    all_descriptors = [l["word"] for l in json.load(data_file)]
keras_model_all = load_model('./models/saved_models/all_descriptors.hdf5')
model_all = TopicDetectionModel(keras_model=keras_model_all, word2vec_model=word2vecmodel, scaler=scaler, labels=all_descriptors)

desc_and_tax = []
with open('./models/descriptors_with_taxonomies.json') as data_file:
    desc_and_tax = json.load(data_file)
keras_model_tax = load_model('./models/saved_models/descriptors_and_taxonomies.hdf5')
model_with_tax = TopicDetectionModel(keras_model=keras_model_tax, word2vec_model=word2vecmodel, scaler=scaler, labels=desc_and_tax)

taxonomies = []
with open('./models/taxonomies_list.json') as data_file:
    taxonomies = json.load(data_file)
keras_model_just_tax = load_model('./models/saved_models/just_taxonomies.hdf5')
model_just_tax = TopicDetectionModel(keras_model=keras_model_just_tax, word2vec_model=word2vecmodel, scaler=scaler, labels=taxonomies)

app = Flask(__name__)


print "Ready"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def dcgan():
    text = request.json["text"]
    res600 = model600.predict(text)
    res3000 = model3000.predict(text)
    res_all = model_all.predict(text)
    res_with_tax = model_with_tax.predict(text)
    res_just_tax = model_just_tax.predict(text)
    return jsonify({'descriptors_600': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res600[:30]]),
                    'descriptors_3000': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res3000[:30]]),
                    'all_descriptors': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_all[:30]]),
                    'descriptors_and_tax': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_with_tax[:30]]),
                    'taxonomies': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_just_tax[:30]]),
                    })

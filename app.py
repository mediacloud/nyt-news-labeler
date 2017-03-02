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

models_dir = "./models"
models_file_name = "saved_models"
models_file_path = os.path.join(models_dir, models_file_name)

if not os.path.isdir(os.path.join(models_file_path)):
    print "Trained models not found, downloading files..."
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/466924777/nyt_labels/saved_models.zip", models_file_path+".zip")
    with zipfile.ZipFile(models_file_path+".zip","r") as zip_ref:
        zip_ref.extractall(models_dir)

labels3000 = []
with open('./models/labels_long.json') as data_file:
  labels3000 = json.load(data_file)
scaler3000 = load_scaler_from_disk('./scaler/scaler_labels_long')
keras_model3000 = load_model('./models/saved_models/labels3000/weights.01-0.00.hdf5')
model3000 = TopicDetectionModel(keras_model=keras_model3000, word2vec_model=word2vecmodel, scaler=scaler3000, labels=labels3000)

labels600 = []
with open('./models/labels.json') as data_file:
  labels600 = json.load(data_file)
scaler600 = load_scaler_from_disk('./scaler/scaler')
keras_model600 = load_model('./models/saved_models/labels600/trained_model.h5')
model600 = TopicDetectionModel(keras_model=keras_model600, word2vec_model=word2vecmodel, scaler=scaler600, labels=labels600)

labels_all = []
with open('./models/descriptors.json') as data_file:
  labels_all = [l["word"] for l in json.load(data_file)]
scaler_all = load_scaler_from_disk('./scaler/scalar_all_labels')
keras_model_all = load_model('./models/saved_models/labels_all/trained_model_all_labels.h5')
model_all = TopicDetectionModel(keras_model=keras_model_all, word2vec_model=word2vecmodel, scaler=scaler_all, labels=labels_all)

labels_tax = []
with open('./models/labels_with_taxonomies.json') as data_file:
  labels_tax = json.load(data_file)
scaler_tax = load_scaler_from_disk('./scaler/scalar_all_labels')
keras_model_tax = load_model('./models/saved_models/labels_taxonomies/weights.01-0.00.hdf5')
model_tax = TopicDetectionModel(keras_model=keras_model_tax, word2vec_model=word2vecmodel, scaler=scaler_tax, labels=labels_tax)

app = Flask(__name__)
# for debugging puerposes
app.config['TEMPLATES_AUTO_RELOAD'] = True


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
    res_tax = model_tax.predict(text)
    return jsonify({'labels600': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res600[:30]]),
                    'labels3000': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res3000[:30]]),
                    'labels_all': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_all[:30]]),
                    'labels_tax': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_tax[:30]]),
                    })

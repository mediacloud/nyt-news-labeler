from flask import Flask, request, jsonify, render_template
from magpie import MagpieModel
from keras.models import load_model
from magpie.utils import load_from_disk, save_to_disk
import gensim
import json
import os
import urllib
import zipfile


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
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/466924777/nyt_labels/saved_models.zip", models_dir)

with zipfile.ZipFile(models_file_path+".zip","r") as zip_ref:
    zip_ref.extractall()




labels3000 = []
with open('./models/labels_long.json') as data_file:
  labels3000 = json.load(data_file)
scaler3000 = load_from_disk('./scaler/scaler_labels_long')
keras_model3000 = load_model('./models/saved_models/labels3000/weights.01-0.00.hdf5')
model3000 = MagpieModel(keras_model=keras_model3000, word2vec_model=word2vecmodel, scaler=scaler3000, labels=labels3000)

labels600 = []
with open('./models/labels.json') as data_file:
  labels600 = json.load(data_file)
scaler600 = load_from_disk('./scaler/scaler')
keras_model600 = load_model('./models/saved_models/labels_600/trained_model.h5')
model600 = MagpieModel(keras_model=keras_model600, word2vec_model=word2vecmodel, scaler=scaler600, labels=labels600)

labels_all = []
with open('./models/descriptors.json') as data_file:
  labels_all = [l["word"] for l in json.load(data_file)]
scaler_all = load_from_disk('./scaler/scalar_all_labels')
keras_model_all = load_model('./models/saved_models/labels_all/trained_model_all_labels.h5')
model_all = MagpieModel(keras_model=keras_model_all, word2vec_model=word2vecmodel, scaler=scaler_all, labels=labels_all)

labels_tax = []
with open('./models/labels_with_taxonomies.json') as data_file:
  labels_tax = json.load(data_file)
scaler_tax = load_from_disk('./scaler/scalar_all_labels')
keras_model_tax = load_model('./models/saved_models/labels_taxonomies/weights.01-0.00.hdf5')
model_tax = MagpieModel(keras_model=keras_model_tax, word2vec_model=word2vecmodel, scaler=scaler_tax, labels=labels_tax)

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
    res600 = model600.predict_from_text(text)
    res3000 = model3000.predict_from_text(text)
    res_all = model_all.predict_from_text(text)
    res_tax = model_tax.predict_from_text(text)
    return jsonify({'labels600': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res600[:30]]),
                    'labels3000': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res3000[:30]]),
                    'labels_all': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_all[:30]]),
                    'labels_tax': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_tax[:30]]),
                    })

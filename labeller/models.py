import os
import logging
import gensim
from keras.models import load_model
import json
import sys
import cPickle as pickle

from labeller import base_dir
from labeller.topicdetection import TopicDetectionModel, VectorizerModel

logger = logging.getLogger(__name__)

# global model objects
word2vecmodel = None
model3000 = None
model600 = None
model_all = None
model_with_tax = None
model_just_tax = None
vectorize_model = None


def initialize():
    logger.info("Prepping models:")
    if not _vectors_file_exists():
        logger.error("Missing Google News models! Follow the README.md instructions for downloading and installing them")
        sys.exit()
    _load_vectors_file()
    _load_scalers()


def _path_to_vectors_file():
    dir_name = "word2vec-GoogleNews-vectors"
    file_name = "GoogleNews-vectors-negative300.unit_normalized.bin"
    return os.path.join(base_dir, dir_name, file_name)


def _vectors_file_exists():
    return os.path.exists(_path_to_vectors_file()) and os.path.exists(_path_to_vectors_file() + '.vectors.npy')


def _load_vectors_file():
    global word2vecmodel
    logger.info("Loading pre-trained word2vec model...")
    word2vecmodel = gensim.models.KeyedVectors.load(_path_to_vectors_file(), mmap='r')

    # https://stackoverflow.com/a/43067907
    logger.info("  Preventing recalculation of normalized vectors...")
    word2vecmodel.syn0norm = word2vecmodel.syn0

    logger.info("  Paging model into memory...")
    word2vecmodel.most_similar('news')

    logger.info("Loaded pre-trained word2vec model.")


def _load_scaler_to_memory(path):
    """ Load a pickle from disk to memory """
    if not os.path.exists(path):
        raise ValueError("File " + path + " does not exist")
    return pickle.load(open(path, 'rb'))


def _load_scalers():
    global model3000, model600, model_all, model_with_tax, model_just_tax,vectorize_model

    models_dir = os.path.join(base_dir, "models")
    saved_models_dir = os.path.join(models_dir, "saved_models")

    # load pre-trained scaler used by all the models
    scaler = _load_scaler_to_memory(os.path.join(base_dir, "scaler", "scaler"))

    data_file = open(os.path.join(models_dir, 'descriptors_3000.json'))
    descriptors_3000 = json.load(data_file)
    keras_model3000 = load_model(os.path.join(saved_models_dir, 'descriptors_3000.hdf5'))
    model3000 = TopicDetectionModel(keras_model=keras_model3000, word2vec_model=word2vecmodel,
                                    scaler=scaler, labels=descriptors_3000)

    data_file = open(os.path.join(models_dir, 'descriptors_600.json'))
    descriptors_600 = json.load(data_file)
    keras_model600 = load_model(os.path.join(saved_models_dir, 'descriptors_600.hdf5'))
    model600 = TopicDetectionModel(keras_model=keras_model600, word2vec_model=word2vecmodel,
                                   scaler=scaler, labels=descriptors_600)

    data_file = open(os.path.join(models_dir, 'all_descriptors.json'))
    all_descriptors = [l["word"] for l in json.load(data_file)]
    keras_model_all = load_model(os.path.join(saved_models_dir, 'all_descriptors.hdf5'))
    model_all = TopicDetectionModel(keras_model=keras_model_all, word2vec_model=word2vecmodel,
                                    scaler=scaler, labels=all_descriptors)

    data_file = open(os.path.join(models_dir, 'descriptors_with_taxonomies.json'))
    desc_and_tax = json.load(data_file)
    keras_model_tax = load_model(os.path.join(saved_models_dir, 'descriptors_and_taxonomies.hdf5'))
    model_with_tax = TopicDetectionModel(keras_model=keras_model_tax, word2vec_model=word2vecmodel,
                                         scaler=scaler, labels=desc_and_tax)

    data_file = open(os.path.join(models_dir, 'taxonomies_list.json'))
    taxonomies = json.load(data_file)
    keras_model_just_tax = load_model(os.path.join(saved_models_dir, 'just_taxonomies.hdf5'))
    model_just_tax = TopicDetectionModel(keras_model=keras_model_just_tax, word2vec_model=word2vecmodel,
                                         scaler=scaler, labels=taxonomies)

    vectorize_model = VectorizerModel(word2vec_model=word2vecmodel, scaler=scaler)

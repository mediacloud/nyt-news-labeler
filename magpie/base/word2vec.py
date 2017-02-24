import os

import numpy as np

from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

from magpie.base.document import Document
from magpie.config import WORD2VEC_MODELPATH, EMBEDDING_SIZE
from magpie.utils import get_documents, save_to_disk, get_documents_from_mongo



def compute_word2vec_for_phrase(phrase, model):
    """
    Compute (add) word embedding for a multiword phrase using a given model
    :param phrase: unicode, parsed label of a keyphrase
    :param model: gensim word2vec object

    :return: numpy array
    """
    result = np.zeros(model.vector_size, dtype='float32')
    for word in phrase.split():
        if word in model:
            result += model[word]

    return result

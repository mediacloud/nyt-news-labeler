import numpy as np
from nltk.tokenize import word_tokenize

PUNCTUATION = '.,:;!?()/\"-<>[]{}|\\@#`$%^&*'


class TopicDetectionModel:
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

        zipped = list(zip(self.labels, y_predicted[0]))

        return sorted(zipped, key=lambda elem: elem[1], reverse=True)


class VectorizerModel:
    def __init__(self, word2vec_model, scaler):
        self.word2vec_model = word2vec_model
        self.scaler = scaler

    def vectorize(self, text, sample_length=200, embedding_size=300):

        words = [w.lower() for w in word_tokenize(text)
                 if w not in PUNCTUATION][:sample_length]
        x_matrix = np.zeros((1, sample_length, embedding_size))

        for i, w in enumerate(words):
            if w in self.word2vec_model:
                word_vector = self.word2vec_model[w].reshape(1, -1)
                scaled_vector = self.scaler.transform(word_vector, copy=True)[0]
                x_matrix[0][i] = scaled_vector
        return x_matrix.tolist()

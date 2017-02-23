import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(ROOT_DIR, 'data', 'hep-keywords')

# word2vec & scaler
EMBEDDING_SIZE = 300
WORD2VEC_MODELPATH = os.path.join(
    CORPUS_DIR,
    'w2v_models',
    'word2vec{0}_model.gensim'.format(EMBEDDING_SIZE)
)
SCALER_PATH = os.path.join(
    CORPUS_DIR,
    'scalers',
    'scaler_nn_{0}.pickle'.format(EMBEDDING_SIZE)
)

# Models
NN_ARCHITECTURE = 'cnn'

# Training parameters
BATCH_SIZE = 128
NB_EPOCHS = 1

# Number of tokens to save from the abstract, zero padded
SAMPLE_LENGTH = 200

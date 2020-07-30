from flask import Flask, request, jsonify, render_template
import logging
import datetime as dt
from typing import Dict, List

from models import (
    SELF_TEST_INPUT,
    Descriptors600Model,
    Word2vecModel,
    Descriptors3000Model,
    DescriptorsAllModel,
    DescriptorsWithTaxonomiesModel,
    JustTaxonomiesModel,
    Scaler)

VERSION = "1.1.0"

logger = logging.getLogger(__name__)
logger.info("---------------------------------------------------------------------------")

# Load the models into memory and validate that they are working
logger.info("Intializing...")
logger.info("  Loading models...")
word2vec_model = Word2vecModel()
scaler = Scaler()
MODEL_600 = Descriptors600Model(word2vec_model=word2vec_model, scaler=scaler)
MODEL_3000 = Descriptors3000Model(word2vec_model=word2vec_model, scaler=scaler)
MODEL_ALL = DescriptorsAllModel(word2vec_model=word2vec_model, scaler=scaler)
MODEL_WITH_TAX = DescriptorsWithTaxonomiesModel(word2vec_model=word2vec_model, scaler=scaler)
MODEL_JUST_TAX = JustTaxonomiesModel(word2vec_model=word2vec_model, scaler=scaler)
logger.info("  done loading models...")

logger.info("  Running self-test...")
for model in [MODEL_600, MODEL_3000, MODEL_ALL, MODEL_WITH_TAX, MODEL_JUST_TAX]:
    logger.debug("Model %s:" % model.__class__.__name__)
    test_predictions = model.predict(SELF_TEST_INPUT)
    for prediction in test_predictions:
        logger.debug("  * Label: %s, score: %2.6f" % (prediction.label, prediction.score,))
    assert len(test_predictions), "Some predictions should be returned by %s" % model.__class__.__name__
logger.info("  done running self-test.")

logger.info("  Starting web app...")
app = Flask(__name__)
logger.info("  ready for requests.")
logger.info("Intialization done!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict.json', methods=['POST'])
def predict():
    start = dt.datetime.now()
    text = request.json["text"]
    try:
        predictions = _predict(text)
        results = {
            'status': 'OK',
            'predictions': predictions,
        }
    except Exception as e:
        results = {
            'status': 'FAILED',
            'exception': str(e),
        }
    end = dt.datetime.now()
    results['milliseconds'] = (end - start).total_seconds() * 1000
    results['version'] = VERSION
    return jsonify(results)


@app.route('/word2vec', methods=['POST'])
def word2vec():
    text = request.json["text"]
    result = word2vec_model.predict(text)
    return jsonify(result)


def _predict(text: str) -> Dict[str, List[Dict[str, str]]]:
    global MODEL_600
    global MODEL_3000
    global MODEL_ALL
    global MODEL_WITH_TAX
    global MODEL_JUST_TAX
    result_600 = MODEL_600.predict(text)
    result_3000 = MODEL_3000.predict(text)
    result_all = MODEL_ALL.predict(text)
    result_with_tax = MODEL_WITH_TAX.predict(text)
    result_just_tax = MODEL_JUST_TAX.predict(text)
    results = {
        'descriptors600': [
            {'label': x.label, 'score': "{0:.5f}".format(x.score)} for x in result_600
        ],
        'descriptors3000': [
            {'label': x.label, 'score': "{0:.5f}".format(x.score)} for x in result_3000
        ],
        'allDescriptors': [
            {'label': x.label, 'score': "{0:.5f}".format(x.score)} for x in result_all
        ],
        'descriptorsAndTaxonomies': [
            {'label': x.label, 'score': "{0:.5f}".format(x.score)} for x in result_with_tax
        ],
        'taxonomies': [
            {'label': x.label, 'score': "{0:.5f}".format(x.score)} for x in result_just_tax
        ],
    }
    return results

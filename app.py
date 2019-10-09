from flask import Flask, request, jsonify, render_template
import logging
from labeller import models

logger = logging.getLogger(__name__)
logger.info("---------------------------------------------------------------------------")

# load up all the models into memory
models.initialize()

logger.info("Starting web app")
app = Flask(__name__)

VERSION = '1.1.1';

# When an exception gets raised, log it and don't just quietly shut down
app.config['PROPAGATE_EXCEPTIONS'] = True


@app.route('/')
def index():
    logger.debug("Request to homepage")
    return render_template('index.html')


@app.route('/predict.json', methods=['POST'])
def predict():
    logger.debug("Request to predict.json")
    text = request.json["text"]
    res600 = models.model600.predict(text)
    res3000 = models.model3000.predict(text)
    res_all = models.model_all.predict(text)
    res_with_tax = models.model_with_tax.predict(text)
    res_just_tax = models.model_just_tax.predict(text)
    return jsonify({'descriptors600': [_format_result(x) for x in res600[:30]],
                    'descriptors3000': [_format_result(x) for x in res3000[:30]],
                    'allDescriptors': [_format_result(x) for x in res_all[:30]],
                    'descriptorsAndTaxonomies': [_format_result(x) for x in res_with_tax[:30]],
                    'taxonomies': [_format_result(x) for x in res_just_tax[:30]],
                    'version': VERSION,
                    })


def _format_result(x):
    return {'label': x[0], 'score': "{0:.5f}".format(x[1])}


@app.route('/predict', methods=['POST'])
def dcgan():
    text = request.json["text"]
    res600 = models.model600.predict(text)
    res3000 = models.model3000.predict(text)
    res_all = models.model_all.predict(text)
    res_with_tax = models.model_with_tax.predict(text)
    res_just_tax = models.model_just_tax.predict(text)
    return jsonify({'descriptors_600': "\n".join([_format_text(x) for x in res600[:30]]),
                    'descriptors_3000': "\n".join([_format_text(x) for x in res3000[:30]]),
                    'all_descriptors': "\n".join([_format_text(x) for x in res_all[:30]]),
                    'descriptors_and_tax': "\n".join([_format_text(x) for x in res_with_tax[:30]]),
                    'taxonomies': "\n".join([_format_text(x) for x in res_just_tax[:30]]),
                    'version': VERSION,
                    })


@app.route('/word2vec', methods=['POST'])
def word2vec():
    text = request.json["text"]
    result = models.vectorize_model.vectorize(text)
    return jsonify(result)


def _format_text(x):
    return "%s : %s" % (x[0], "{0:.5f}".format(x[1]))

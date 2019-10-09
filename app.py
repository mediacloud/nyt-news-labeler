from flask import Flask, request, jsonify, render_template
import logging
from labeller import models

logger = logging.getLogger(__name__)
logger.info("---------------------------------------------------------------------------")

# load up all the models into memory
models.initialize()

logger.info("Starting web app")
app = Flask(__name__)

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
    return jsonify({'descriptors600': [{'label': x[0], 'score': "{0:.5f}".format(x[1])} for x in res600[:30]],
                    'descriptors3000': [{'label': x[0], 'score': "{0:.5f}".format(x[1])} for x in res3000[:30]],
                    'allDescriptors': [{'label': x[0], 'score': "{0:.5f}".format(x[1])} for x in res_all[:30]],
                    'descriptorsAndTaxonomies': [{'label': x[0], 'score': "{0:.5f}".format(x[1])} for x in res_with_tax[:30]],
                    'taxonomies': [{'label': x[0], 'score': "{0:.5f}".format(x[1])} for x in res_just_tax[:30]],
                    })


@app.route('/predict', methods=['POST'])
def dcgan():
    text = request.json["text"]
    res600 = models.model600.predict(text)
    res3000 = models.model3000.predict(text)
    res_all = models.model_all.predict(text)
    res_with_tax = models.model_with_tax.predict(text)
    res_just_tax = models.model_just_tax.predict(text)
    return jsonify({'descriptors_600': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res600[:30]]),
                    'descriptors_3000': "\n".join(["%s : %s"%(x[0], "{0:.5f}".format(x[1])) for x in res3000[:30]]),
                    'all_descriptors': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_all[:30]]),
                    'descriptors_and_tax': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_with_tax[:30]]),
                    'taxonomies': "\n".join(["%s : %s" % (x[0], "{0:.5f}".format(x[1])) for x in res_just_tax[:30]]),
                    })

@app.route('/word2vec', methods=['POST'])
def word2vec():
    text =  request.json["text"]
    result = models.vectorize_model.vectorize(text)
    return jsonify(result)


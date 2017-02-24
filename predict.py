#!/usr/bin/python
# -*- coding: utf-8 -*-
from magpie import MagpieModel
from keras.models import load_model
from magpie.utils import load_from_disk, save_to_disk
import gensim
import json

print "Loading pre-trained word to vec model"
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
print "model_loaded"
labels = []
with open('./models/labels_long.json') as data_file:
  labels = json.load(data_file)
scaler = load_from_disk('./scaler/scaler_labels_long')
keras_model = load_model('./models/weights.00-0.00.hdf5')

model = MagpieModel(keras_model=keras_model, word2vec_model=word2vecmodel, scaler=scaler, labels=labels)
#save_to_disk('./scalar/embeddings', model.word2vec_model)
print "predicting...!"

res = model.predict_from_text(u'''Gold futures are headed for the longest streak of losses since November as buying from China stops ahead of its week-long holiday to celebrate the Lunar New Year.

Prices dropped Friday for the fourth straight day, cutting the year’s gains. China, the biggest consumer of gold, increased gold purchases in the run-up to the start of the Year of the Rooster this week, when bars or jewelry made from the metal are traditionally given as gifts.


"The Chinese holiday can exaggerate some of the moves," Bob Haberkorn, a senior market strategist at RJO Futures in Chicago, said in a telephone interview. "We’re going to get lighter volume coming in. A lot of the focus is moving into risk assets."

Gold futures for April delivery fell 0.1 percent to $1,191.40 at 11:08 a.m. on the Comex in New York. Futures earlier fell as much as 0.8 percent, touching the lowest since Jan. 11. The four-day losing streak would be the longest since Nov. 14.

After touching a two-month high earlier this week, gold’s rally has withered as surging stock markets fueled investor appetite for risk. The Dow Jones Industrial Average climbed above 20,000 for the first time this week and the MSCI All-Country World Index is near a record.

The metal pared earlier losses Friday after a report showed U.S. economic growth slowed more than forecast last quarter on the biggest drag from trade in six years and more moderate consumer spending. Business investment picked up, which may be a harbinger for faster expansion in 2017.

“We expect full year Chinese demand to still fall short of levels seen in 2015,” Nell Agate, a Citigroup Inc. metals analyst, said by e-mail from London. “It’s possible that Chinese jewelry sales are likely to slow as the Chinese break for new year festivities.”

For more on gold’s recent price performance, click here.

In other precious metals:

Silver futures for March delivery rose 1.3 percent to $17.07 an ounce.
Platinum futures for April delivery fell 0.1 percent to $980.70 an ounce.
Palladium futures for March delivery climbed 2.2 percent to $740.20 an ounce. The metal is down 6.2 percent this week.''')

print res[:20]
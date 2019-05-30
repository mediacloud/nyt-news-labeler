#!/usr/bin/env python2.7
#
# Convert Google News model to KeyedVectors and unit-normalize it so that:
#
# * We can load it and share among workers using mmap()
# * It loads faster
# * It uses at least 2x less memory
#
# Please see:
#
#     https://stackoverflow.com/a/43067907
#
# for more details.
#
# After generating the new model, zip both generated files into a single archive:
#
#   zip -9 GoogleNews-vectors-negative300.unit_normalized.bin.zip \
#       GoogleNews-vectors-negative300.unit_normalized.bin \
#       GoogleNews-vectors-negative300.unit_normalized.bin.vectors.npy
#
# Lastly, upload the archive to S3 and update download_models.py accordingly.
#

from gensim import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.init_sims(replace=True)
model.save('GoogleNews-vectors-negative300.unit_normalized.bin')

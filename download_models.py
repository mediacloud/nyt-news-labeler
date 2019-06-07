#!/usr/bin/env python2.7
#
# Makes sure all the required models are downloaded and extracted
#

import os
import subprocess


def __pwd():
    """Return full path to the script's directory."""
    return os.path.dirname(os.path.realpath(__file__))


def __download_file(url, dest_path):
    """Download file to target path."""

    # System cURL is way faster than Python's requests at downloading huge files
    args = [
        "curl",
        # "--silent",
        "--continue-at", "-",
        "--show-error",
        "--fail",
        "--retry", "3",
        "--retry-delay", "5",
        "--output", dest_path,
        url
    ]
    subprocess.check_call(args)


def __decompress_file(brotli_file):
    """Decompress Brotli file to destination directory."""

    args = ["brotli", "-d", brotli_file]
    subprocess.check_call(args)


def download_model(url, dest_dir, expected_size):
    """Download model from URL to a specified destination directory, check if the size is correct, decompress."""

    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)

    # File that gets created for every model when its downloaded and extracted successfully
    downloaded_marker_file = ".%s.downloaded" % (filename,)
    downloaded_marker_path = os.path.join(dest_dir, downloaded_marker_file)

    if os.path.exists(downloaded_marker_path):
        print("Model %s already exists." % (dest_path,))

    else:

        need_to_download = True
        if os.path.isfile(dest_path):
            if os.path.getsize(dest_path) != expected_size:
                print("Found a partial download, will continue it at %s..." % (dest_path,))
            else:
                need_to_download = False
        else:
            print("Model %s was not found, will start download from %s..." % (dest_path, url,))

        if need_to_download:
            __download_file(url=url, dest_path=dest_path)

        if os.path.getsize(dest_path) != expected_size:
            raise Exception("Downloaded file size is not %d." % (expected_size,))

        print("Decompressing %s to %s..." % (dest_path, dest_dir,))
        __decompress_file(brotli_file=dest_path)

        os.unlink(dest_path)

        open(downloaded_marker_path, 'a').close()


def download_all_models():
    """Download and prepare all the required models."""

    models = [
        # See word2vec_to_keyedvectors.py
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/GoogleNews-vectors-negative300.keyedvectors.bin.br',
            'dest_dir': __pwd() + '/word2vec-GoogleNews-vectors/',
            'expected_size': 68284073,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/GoogleNews-vectors-negative300.keyedvectors.bin.vectors.npy.br',
            'dest_dir': __pwd() + '/word2vec-GoogleNews-vectors/',
            'expected_size': 1316205343,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/all_descriptors.hdf5.br',
            'dest_dir': __pwd() + '/models/saved_models/',
            'expected_size': 370734856,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/descriptors_3000.hdf5.br',
            'dest_dir': __pwd() + '/models/saved_models/',
            'expected_size': 61285705,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/descriptors_600.hdf5.br',
            'dest_dir': __pwd() + '/models/saved_models/',
            'expected_size': 21018967,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/descriptors_and_taxonomies.hdf5.br',
            'dest_dir': __pwd() + '/models/saved_models/',
            'expected_size': 95996935,
        },
        {
            'url': 'https://mediacloud-nytlabels-data.s3.amazonaws.com/predict-news-labels-keyedvectors/just_taxonomies.hdf5.br',
            'dest_dir': __pwd() + '/models/saved_models/',
            'expected_size': 51423506,
        },
    ]

    for model in models:
        download_model(url=model['url'], dest_dir=model['dest_dir'], expected_size=model['expected_size'])


if __name__ == '__main__':
    download_all_models()

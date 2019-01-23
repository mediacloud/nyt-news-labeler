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
        "--show-error",
        "--fail",
        "--retry", "3",
        "--retry-delay", "5",
        "--output", dest_path,
        url
    ]
    subprocess.check_call(args)


def __unzip_file(zip_file, dest_dir):
    """Unzip file to destination directory."""

    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    # System's `unzip` is also faster than Python's "zipfile"
    args = ["unzip", "-o", zip_file]
    subprocess.check_call(args, cwd=dest_dir)


def download_model(url, dest_dir, expected_size):
    """Download model from URL to a specified destination directory, check if the size is correct, unzip."""

    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)

    # File that gets created for every model when its downloaded and extracted successfully
    downloaded_marker_file = ".%s.downloaded" % (filename,)
    downloaded_marker_path = os.path.join(dest_dir, downloaded_marker_file)

    if os.path.exists(downloaded_marker_path):
        print "Model %s already exists." % (dest_path,)

    else:

        need_to_download = True
        if os.path.isfile(dest_path):
            if os.path.getsize(dest_path) != expected_size:
                print "Removing incomplete model file %s..." % (dest_path,)
                os.unlink(dest_path)
            else:
                need_to_download = False

        if need_to_download:
            print "Model %s was not found, downloading from %s..." % (dest_path, url,)
            __download_file(url=url, dest_path=dest_path)

        if os.path.getsize(dest_path) != expected_size:
            raise Exception("Downloaded file size is not %d." % (expected_size,))

        print "Unzipping %s to %s..." % (dest_path, dest_dir,)
        __unzip_file(zip_file=dest_path, dest_dir=dest_dir)

        os.unlink(dest_path)

        open(downloaded_marker_path, 'a').close()


def download_all_models():
    """Download and prepare all the required models."""

    models = [
        {
            'url': 'https://s3.amazonaws.com/mediacloud-nytlabels-data/predict-news-labels-repackaged/'
                   'GoogleNews-vectors-negative300.bin.zip',
            'dest_dir': __pwd() + '/word2vec-GoogleNews-vectors/',
            'expected_size': 1647046392,
        },
        {
            'url': 'https://s3.amazonaws.com/mediacloud-nytlabels-data/predict-news-labels-repackaged/saved_models.zip',
            'dest_dir': __pwd() + '/models/',
            'expected_size': 616582073,
        }
    ]

    for model in models:
        download_model(url=model['url'], dest_dir=model['dest_dir'], expected_size=model['expected_size'])


if __name__ == '__main__':
    download_all_models()

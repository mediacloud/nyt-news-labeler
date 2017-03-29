import os
import urllib

emmbedings_dir = "./word2vec-GoogleNews-vectors"
emmbedings_file_name = "GoogleNews-vectors-negative300.bin"
emmbedings_file_path = os.path.join(emmbedings_dir, emmbedings_file_name)

if not os.path.exists(emmbedings_dir):
    os.mkdir(emmbedings_dir)
if not os.path.isfile(emmbedings_file_path):
    print "Google word2vec model not found, downloading file..."
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/466924777/GoogleNews-vectors-negative300.bin", emmbedings_file_path)
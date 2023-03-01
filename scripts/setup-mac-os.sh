# notes via https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64

## create empty environment
conda create -n py37

## activate
conda activate py37

## use x86_64 architecture channel(s)
conda config --env --set subdir osx-64

## install python, numpy, etc. (add more packages here...)
conda install python=3.7

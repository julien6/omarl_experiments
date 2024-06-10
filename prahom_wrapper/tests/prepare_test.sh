#!/bin/bash

python3.9 -m venv prahom_test;
source prahom_test/bin/activate;

if ! test -f ./../../custom_envs/dist/*.tar.gz; then
    curr_loc=$(pwd)
    cd ./../../custom_envs/
    python setup.py sdist;
    cd $curr_loc
fi

pip install protobuf==3.20.*

pip install -r ../requirements.txt

python ./prahom_test/lib/python3.9/site-packages/marllib/patch/add_patch.py -y

deactivate

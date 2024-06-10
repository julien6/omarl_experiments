#!/bin/bash

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9 -y
sudo apt-get install python3.9-venv -y


python3.9 -m venv prahom_test;
source prahom_test/bin/activate;

if ! test -f ./../../custom_envs/dist/*.tar.gz; then
    curr_loc=$(pwd)
    cd ./../../custom_envs/
    python setup.py sdist;
    cd $curr_loc
fi

pip install -r ../requirements.txt

pip install protobuf==3.20.*

python ./prahom_test/lib/python3.9/site-packages/marllib/patch/add_patch.py -y

deactivate

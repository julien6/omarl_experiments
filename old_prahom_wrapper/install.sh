#!/bin/bash

if ! test -f ./../custom_envs/dist/*.tar.gz; then
    curr_loc=$(pwd)
    cd ./../custom_envs/
    echo $curr_loc
    python setup.py sdist;
    cd $curr_loc
fi

python -m venv venv;
source venv/bin/activate;
python -m pip install -r requirements.txt --timeout 86400;

pip install stable-baselines3[extra];
pip install tensorboard;
pip install ./../custom_envs/dist/*.tar.gz;

echo "Use the following command to launch mcy simulation with the proper python environment";
echo "'source ./venv/bin/activate'"

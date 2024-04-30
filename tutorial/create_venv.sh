#!/bin/bash

python -m venv tuto_env;
source tuto_env/bin/activate;
python -m pip install ipykernel;
python -m ipykernel install --user --name=tuto_env;

if ! test -f ./../custom_envs/dist/*.tar.gz; then
    curr_loc=$(pwd)
    cd ./../custom_envs/
    python setup.py sdist;
    cd $curr_loc
fi

deactivate
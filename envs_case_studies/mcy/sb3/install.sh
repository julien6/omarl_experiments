#!/bin/bash
mkdir python-envs;
cd python-envs;
python -m venv mcy;
source mcy/bin/activate;
cd ..;
python -m pip install -r requirements.txt --timeout 86400;
pip install stable-baselines3[extra];
pip install tensorboard;
echo "Use the following command to launch mcy simulation with the proper python environment";
echo "'source ./python-envs/mcy/bin/activate'"

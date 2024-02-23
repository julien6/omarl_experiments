#!/bin/bash
cwd=$(pwd);
cd ~;
mkdir python-envs;
cd python-envs;
python -m venv pistonball;
source pistonball/bin/activate;
cd $cwd;
pip install -r requirements.txt;
echo "Use the following command to launch pistonball simulation with the proper python environment";
echo "'source ~/python-envs/pistonball/bin/activate'"
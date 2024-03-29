#!/bin/bash
mkdir python-envs;
cd python-envs;
python3 -m venv pistonball;
source pistonball/bin/activate;
cd ..;
python3 -m pip install -r requirements.txt --timeout 86400;
python3 -m pip install "ray[tune]" --timeout 86400;
echo "Use the following command to launch pistonball simulation with the proper python environment";
echo "'source ./python-envs/pistonball/bin/activate'"

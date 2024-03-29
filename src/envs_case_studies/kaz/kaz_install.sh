#!/bin/bash
mkdir python-envs;
cd python-envs;
python3 -m venv kaz;
source kaz/bin/activate;
cd ..;
python3 -m pip install -r requirements.txt --timeout 86400;
echo "Use the following command to launch KAZ simulation with the proper python environment";
echo "'source ./python-envs/kaz/bin/activate'";
echo "Then run 'python kaz.py' to start the simulation";
#!/bin/bash

#OAR -n MPE
#OAR -l /nodes=1/gpu=1,walltime=3:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

cd /bettik/soulej/omarl_experiments/
source prahom_wrapper/tests/prahom_test/bin/activate
python prahom_wrapper/tests/marllib_basic_use.py
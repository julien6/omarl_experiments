#!/bin/bash

#OAR -n MovingCompany
#OAR -l /nodes=1/gpu=1,walltime=4:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

bash -i -c '

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate marllib

python test.py'

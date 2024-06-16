conda create -n marllib python=3.8 -y

bash -i -c '

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate marllib
conda --version

git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install -r requirements.txt
pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip install -r requirements.txt
pip install "gym>=0.20.0,<0.22.0"
pip install ray[tune]

pip install protobuf==3.20.*
python marllib/patch/add_patch.py -y

pip install pettingzoo==1.23.1
pip install supersuit==3.9.0
pip install pygame==2.3.0

conda install -c conda-forge libstdcxx-ng
pip install pyglet==1.5.11

pip install marllib
'

# oarsub -I -l /nodes=1/gpu=1,walltime=00:30:00 -p "gpumodel='V100'" --project pr-ai4cmas
# oarsub -S ./run_train_test.sh
# source /applis/environments/conda.sh
# conda activate marllib
# python test.py

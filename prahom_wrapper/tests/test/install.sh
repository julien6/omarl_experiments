sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

if [ ! -f "/applis/environments/conda.sh" ] && [ ! -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
    chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
    ./Anaconda3-2024.02-1-Linux-x86_64.sh
fi

    
bash -i -c '

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate marllib
conda --version
conda create -n marllib python=3.8 -y

rm -rf Anaconda3-2024.02-1-Linux-x86_64.sh
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

echo -e "INSTALLATION ON COMPUTATIONAL REMOTE SERVER"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh"]; then
    ssh soulej@bigfoot.ciment -t "cd /bettik/soulej ; git clone https://github.com/julien6/omarl_experiments.git ; cd omarl_experiments ; git checkout test ; cd prahom_wrapper/tests/test ; ./install.sh"
fi
'
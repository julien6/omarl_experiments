#!/bin/bash

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
fi

cat <<EOT > tmp_install.sh
#!/bin/bash

if [ ! -f "/applis/environments/conda.sh" ] && [ ! -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
    chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
    ./Anaconda3-2024.02-1-Linux-x86_64.sh
fi

if [ -e "/applis/environments/conda.sh" ]; then
    source /applis/environments/conda.sh
else
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda --version
conda create -n marllib python=3.8 -y
conda activate marllib

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
EOT

echo -e "\n\nINSTALLATION ON INTERFACE LOCAL MACHINE\n"
chmod +x tmp_install.sh
./tmp_install.sh

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo -e "\n\nINSTALLATION ON COMPUTATIONAL REMOTE SERVER\n"
    ssh soulej@bigfoot.ciment -t "cd /bettik/soulej ; rm -rf omarl_experiments ; mkdir omarl_experiments"
    rsync -avxH tmp_install.sh oar_launch.sh train_test.py soulej@bigfoot.ciment:/bettik/soulej/omarl_experiments/
    rm -rf tmp_install.sh
    ssh soulej@bigfoot.ciment -t "cd /bettik/soulej/omarl_experiments ; ./tmp_install.sh ; rm -rf tmp_install.sh ; echo -e \"\n\nINSTALLATION FINISHED!\n\""
fi

#!/bin/bash

# Vérifiez si Python 3.10 est installé
if ! python3.10 --version &> /dev/null; then
    echo "Python 3.10 n'est pas installé. Veuillez l'installer avant d'exécuter ce script."
    exit 1
fi

# Créez un environnement virtuel basé sur Python 3.10
python3.10 -m venv tuto_env

# Activez l'environnement virtuel
source tuto_env/bin/activate

# Installez ipykernel et d'autres dépendances
python -m pip install --upgrade pip
python -m pip install ipykernel
python -m pip install matplotlib==3.7.0
python -m pip install scikit-learn supersuit stable-baselines3

# Configurez ipykernel pour l'environnement virtuel
python -m ipykernel install --user --name=tuto_env

# Construisez le package dans le répertoire custom_envs
curr_loc=$(pwd)
cd ./../custom_envs/
python setup.py sdist || { echo "Échec de la construction de custom_envs"; exit 1; }
cd $curr_loc

# Construisez le package dans le répertoire mma_wrapper
cd ./../mma_wrapper/
python setup.py sdist || { echo "Échec de la construction de mma_wrapper"; exit 1; }
cd $curr_loc

# Désactivez l'environnement virtuel
deactivate

echo "Installation terminée avec succès !"

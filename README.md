create env

'''
bash

conda create -n wineq python=3.7 spyder=4 -y
activate wineq

git init -b main
git remote add origin

pip install -r requirement.txt

created a template file
    this file contains all the files and directory details of the project folder
    running this file would create data directories, params.yaml, dvc.yaml
    
dvc init
dvc add given_data/winequality-red.csv

created a get_data function that that takes file from given_data folder and stores it in a df



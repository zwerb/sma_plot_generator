#!/bin/bash

echo "Installing Environment for Python3 Jupyterlab Stock Analysis"

sudo pip install --upgrade pip

pip3 install -U jupyter
pip3 install -U jupyterlab

pip3 install yfinance --upgrade
pip3 install numpy --upgrade
pip3 install matplotlib --upgrade
pip3 install seaborn --upgrade
pip3 install pandas --upgrade
pip3 install yahoofinancials --upgrade
pip3 install sklearn --upgrade
pip3 install tensorflow --upgrade
pip3 install keras --upgrade
pip3 install pandas_datareader --upgrade

echo "Updates complete, ready for SMA Generator"
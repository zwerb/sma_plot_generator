#!/usr/bin/env python

#Import the libraries
import math
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')#

import sys

DATADIR = 'stock_files'
GRAPHDIR = 'stock_graphs'
SPLITSDIR = 'split_files'

plt.style.use('seaborn-whitegrid')

import sma_support_functions as sa

if len(sys.argv) == 3:
    ticker = sys.argv[1]
    days_range = sys.argv[2]
    print(sa.online_process_stock_once(str(ticker), int(days_range)))
else:
    print ('invalid number of args. argv=[{}]'.format(str(sys.argv)))
    #ticker = input("Type a ticker: ")
    #days_range = input("Input days: ") 

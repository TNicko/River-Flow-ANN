from cmath import nan
from ftplib import error_perm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random, shuffle
from numpy.linalg import eig
import time, datetime

np.random.seed(2)


# Import dataset
df = pd.read_excel('data.xlsx', header = 0, skiprows=1, usecols="A:I", index_col=0, parse_dates=True)

df.columns = ['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 'Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme']
df[['Skip Bridge','Skelton','East Cowton']] = df[['Skip Bridge','Skelton','East Cowton']].apply(pd.to_numeric, errors='coerce')

# Daily Flow outliers : upper and lower bound checked
df[['Crakehill','Skip Bridge','Westwick', 'Skelton']] = df[['Crakehill','Skip Bridge','Westwick', 'Skelton']].mask(df.sub(df.mean()).div(df.std()).abs().lt(-3))
df[['Crakehill','Skip Bridge','Westwick', 'Skelton']] = df[['Crakehill','Skip Bridge','Westwick', 'Skelton']].mask(df.sub(df.mean()).div(df.std()).abs().gt(3))

# Daily Rainfall outliers : only upper bound checked
df[['Arkengarthdale','East Cowton','Malham Tarn', 'Snaizeholme']] = df[['Arkengarthdale','East Cowton','Malham Tarn', 'Snaizeholme']].mask(df.sub(df.mean()).div(df.std()).abs().gt(3))

# interpolate all NaN types
df[df < 0] = nan
df = df.interpolate()

# Add labels
labels = df['Skelton']
df = df.shift(1)
df['labels'] = labels
df = df.iloc[1:]


# Moving Averages Plots and correlation comparison
width = 3
lag1 = df.shift(1)
lag3 = df[['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 
        'Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme']].shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = pd.concat([means, lag1], axis=1)
dataframe.columns = ['Crakehill ma', 'Skip Bridge ma', 'Westwick ma', 'Skelton ma', 
                     'Arkengarthdale ma', 'East Cowton ma', 'Malham Tarn ma', 'Snaizeholme ma',
                     'Crakehill', 'Skip Bridge', 'Westwick', 'Skelton', 
                     'Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme', 'labels']

dataframe = dataframe.iloc[4:]

flow_corr = dataframe[['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton',
                      'Crakehill ma', 'Skip Bridge ma', 'Westwick ma', 'Skelton ma', 'labels']].corr()['labels'][:-1]

# Drop daily flow moving average columns, and total rainfall original values 
df = dataframe.drop(labels=['Crakehill ma', 'Skip Bridge ma', 'Westwick ma', 'Skelton ma',
                           'Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme'], axis=1)

x_set = df.iloc[:,:-1].to_numpy()

y_set = df['labels'].to_numpy()

dates = df.index.values 

# shuffle data
shuffler = np.random.permutation(len(x_set))
x_set_shuffled = x_set[shuffler]
y_set_shuffled = y_set[shuffler]
dates_shuffled = dates[shuffler]

x_split = np.split(x_set_shuffled, [int(.8 * len(x_set_shuffled))])
y_split = np.split(y_set_shuffled, [int(.8 * len(y_set_shuffled))])
dates_split = np.split(dates_shuffled, [int(.8 * len(dates_shuffled))])

x_train, y_train = x_split[0], y_split[0]
x_test, y_test = x_split[1], y_split[1]

dates_train, dates_test = dates_split[0], dates_split[1]


# standardisation betweeen two values a,b : a + (x - min(x))(b-a) / max(x) - min(x)
# take min,max in train+val set and standardise all sets using these values
a = 0.1
b = 0.9

x_train_min = np.amin(x_train, axis=0)
x_train_max = np.amax(x_train, axis=0)
y_train_min = np.amin(y_train, axis=0)
y_train_max = np.amax(y_train, axis=0)

x_train_norm = a + (x_train-x_train_min)*(b-a)/(x_train_max-x_train_min)
x_test_norm = a + (x_test-x_train_min)*(b-a)/(x_train_max-x_train_min)

y_train_norm = a + (y_train-y_train_min)*(b-a)/(y_train_max-y_train_min)
y_test_norm = a + (y_test-y_train_min)*(b-a)/(y_train_max-y_train_min)
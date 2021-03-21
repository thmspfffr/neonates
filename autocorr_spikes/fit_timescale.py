# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:21:25 2021

@author: mchini
"""

from scipy.io import loadmat
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


folder2load = 'D:/models_neonates/autocorr_spikes/data/'
# see excel file in the repo
exps = pd.read_excel('D:/models_neonates/autocorr_spikes/ExperimentPlanPython.xlsx') 
animals = pd.unique(exps['animal_ID'])

# initialize variables
age = np.zeros(np.shape(animals))
timescale = np.zeros(np.shape(animals))
# whether or not to plot individual fits
to_plot = 0

# define function to extract timescale
def func(lags, A, tau, B):
    return A * (np.exp(-lags/tau) + B)

for idx, animal in enumerate(animals):
    # load autocorr stuff & take 50-500ms window
    ac_stuff = loadmat(folder2load + str(animal) + '.mat')
    autocorr = ac_stuff['auto_corr2fit'].flatten()[:10]
    lags = ac_stuff['lag2fit'].flatten()[:10].astype(float)
    try:
        # fit the curve
        popt, pcov = curve_fit(f=func, xdata=lags,
                               ydata=autocorr, p0=np.r_[0.5, 5, 0.01])
        if to_plot > 0:
            plt.figure()
            plt.plot(lags, autocorr, 'b-', label='data')
            plt.plot(lags, func(lags, *popt), 'g--')
    except RuntimeError:
        popt = np.tile(np.nan, 3)
    # extract age & timescale
    age[idx] = pd.unique(exps['Age'].loc[exps['animal_ID'] == animal])
    timescale[idx] = popt[1]
   

Df = pd.DataFrame()
Df['age']= age
Df['timescale']= timescale
plt.figure()
sns.violinplot(x='age', y='timescale',
               data=Df[Df.age > 2],
               cut=0, scale='width')
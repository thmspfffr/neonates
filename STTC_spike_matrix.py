# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:34:02 2021

@author: mchini
"""

#%% load basic packages
    
import numpy as np
from numba import jit

#%% define elephant STTC

@jit(nopython=True)
def run_P(spiketrain_i, spiketrain_j):
    """
    Check every spike in train 1 to see if there's a spike in train 2
    within 0.005
    """
    N2 = len(spiketrain_j)
    
    # Search spikes of spiketrain_i in spiketrain_j
    # ind will contain index of
    ind = np.searchsorted(spiketrain_j, spiketrain_i)

    # To prevent IndexErrors
    # If a spike of spiketrain_i is after the last spike of spiketrain_j,
    # the index is N2, however spiketrain_j[N2] raises an IndexError.
    # By shifting this index, the spike of spiketrain_i will be compared
    # to the last 2 spikes of spiketrain_j (negligible overhead).
    # Note: Not necessary for index 0 that will be shifted to -1,
    # because spiketrain_j[-1] is valid (additional negligible comparison)
    ind[ind == N2] = N2 - 1

    # Compare to nearest spike in spiketrain_j BEFORE spike in spiketrain_i
    close_left = np.abs(
        spiketrain_j[ind - 1] - spiketrain_i) <= 0.005
    # Compare to nearest spike in spiketrain_j AFTER (or simultaneous)
    # spike in spiketrain_j
    close_right = np.abs(
        spiketrain_j[ind] - spiketrain_i) <= 0.005

    # spiketrain_j spikes that are in [-0.005, 0.005] range of spiketrain_i
    # spikes are counted only ONCE (as per original implementation)
    close = close_left + close_right

    # Count how many spikes in spiketrain_i have a "partner" in
    # spiketrain_j
    return np.count_nonzero(close)
    
@jit(nopython=True)
def run_T(spiketrain, len_rec):
    """
    Calculate the proportion of the total recording time 'tiled' by spikes.
    """
    N = len(spiketrain)
    time_A = 2 * N * 0.005  # maximum possible time

    if N == 1:  # for just one spike in train
        if spiketrain[0] < 0.005:
            time_A += -0.005 + spiketrain[0]
        if spiketrain[0] + 0.005 > len_rec:
            time_A += -0.005 - spiketrain[0] + len_rec
    else:  # if more than one spike in train
        # Vectorized loop of spike time differences
        diff = np.diff(spiketrain)
        diff_overlap = diff[diff < 2 * 0.005]
        # Subtract overlap
        time_A += -2 * 0.005 * len(diff_overlap) + np.sum(diff_overlap)

        # check if spikes are within 0.005 of the start and/or end
        # if so subtract overlap of first and/or last spike
        if spiketrain[0] < 0.005:
            time_A += spiketrain[0] - 0.005

        if (len_rec - spiketrain[N - 1]) < 0.005:
            time_A += -spiketrain[-1] - 0.005 + len_rec

    T = time_A / len_rec
    return T
    
@jit(forceobj=True, parallel=True)
def matrix_spike_time_tiling_coefficient(spike_matrix):
    """
    Calculates the Spike Time Tiling Coefficient (STTC) for an entire spike
    matrix. Uses numba to speed up computations.
    It assumes the spike matrix to have 1ms resolution
    """
    
    # length of recording (assumes spike matrix has 1ms resolution)
    len_rec = np.shape(spike_matrix)[1] / 1000
    num_spike_trains = np.shape(spike_matrix)[0]
    # initialize STTC
    STTC = np.zeros(int(num_spike_trains * (num_spike_trains - 1) / 2))
    idx = 0
    # loop over spike pairs
    for i in range(num_spike_trains):
        # extract spike times
        spiketrain_i = np.array(np.nonzero(spike_matrix[i, :])[0]) / 1000
        for j in range(i + 1, num_spike_trains):
            # extract spike times
            spiketrain_j = np.array(np.nonzero(spike_matrix[i, :])[0]) / 1000
            N1 = len(spiketrain_i)
            N2 = len(spiketrain_j)
        
            if N1 == 0 or N2 == 0:
                index = np.nan
            else:
                TA = run_T(spiketrain_i, len_rec)
                TB = run_T(spiketrain_j, len_rec)
                PA = run_P(spiketrain_i, spiketrain_j)
                PA = PA / N1
                PB = run_P(spiketrain_j, spiketrain_i)
                PB = PB / N2
                # check if the P and T values are 1 to avoid division by zero
                # This only happens for TA = PB = 1 and/or TB = PA = 1,
                # which leads to 0/0 in the calculation of the index.
                # In those cases, every spike in the train with P = 1
                # is within 0.005 of a spike in the other train,
                # so we set the respective (partial) index to 1.
                if PA * TB == 1:
                    if PB * TA == 1:
                        index = 1.
                    else:
                        index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
                elif PB * TA == 1:
                    index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
                else:
                    index = 0.5 * (PA - TB) / (1 - PA * TB) + \
                    0.5 * (PB - TA) / (1 - PB * TA)
            STTC[idx] = (index)
            idx += 1
    return STTC
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:41:57 2018

@author: gilsonmatthieu

A toy model to compare several models of (univariate) time series: gausianly distributed or not, stationary versus non-stationary, with temporal correlations.
The goal is to see how the measures such as kurtosis and autocovariance reflect those differences.
"""

import os
import numpy as np
import scipy.stats as stt
import matplotlib.pyplot as pp

# create directory for results
work_dir = 'ex_stationary_gaussianity/'
if not os.path.exists(work_dir):
    os.mkdir(work_dir)

# script

T = 200 # number of observed time samples
n_rep = 10 # number of repetitions for the simulation

# generation of time series

# normal distribution (white noise, no temporal correlations)
ts_norm = np.random.randn(n_rep,T)

# log-normal distribution (white noise, no temporal correlations)
ts_logn = np.random.lognormal(sigma=0.3,size=[n_rep,T])

# autoregressive process with coefficient a
a = 0.3
ts_ar = np.random.randn(n_rep,T)
ts_ar[:,1:T] += a * ts_ar[:,0:T-1]

# bimodal distribution with random Poisson jumps (with rate r), where the mean is increased by m
m = 1.5
r = 10
ts_nstat = np.random.randn(n_rep,T)
n_jumps = 0
for i_rep in range(n_rep):
    t_on = 0
    t_off = 0
    while t_off<T:
        t_on = min(int(t_off-r*np.log(np.random.rand())),T) # start of up-state (exponential distribution for time till jump)
        t_off = min(int(t_on-r*np.log(np.random.rand())),T) # end of up-state (exponential distribution for time till jump)
        ts_nstat[i_rep,t_on:t_off] += m
        n_jumps += 1
print('average number of jumps:',n_jumps/n_rep)

# pooling time series
n_type = 4
all_ts = np.zeros([n_type,n_rep,T])
all_ts[0,:,:] = ts_norm
all_ts[1,:,:] = ts_logn
all_ts[2,:,:] = ts_ar
all_ts[3,:,:] = ts_nstat

labels = ['norm','logn','ar','n-stat']


# examples of time series
pp.figure(figsize=[6,6])
for i_type in range(n_type):
    pp.axes([0.15,0.8-0.23*i_type,0.8,0.18])
    pp.plot(range(T),all_ts[i_type,0,:])
    pp.axis(xmin=0,xmax=min(T,500))
    if i_type==0:
        pp.xlabel('time')
    pp.ylabel(labels[i_type])
pp.savefig(work_dir+'ex_time_series')
pp.close()


# autocovariance
n_tau = 4
ac = np.zeros([n_type,n_rep,n_tau])
for i_type in range(n_type):
    for i_rep in range(n_rep):
        ts_tmp = all_ts[i_type,i_rep,:]
        ts_tmp -= ts_tmp.mean()
        for i_tau in range(n_tau):
            ac[i_type,i_rep,i_tau] = np.dot(ts_tmp[0:T-n_tau+1],ts_tmp[i_tau:T-n_tau+1+i_tau])/(T-n_tau)

pp.figure(figsize=[6,6])
for i_type in range(n_type):
    pp.axes([0.15,0.8-0.23*i_type,0.8,0.18])
    pp.plot([0,n_tau-1],[0,0],'--k')
    pp.errorbar(range(n_tau),ac[i_type,:,:].mean(0),yerr=ac[i_type,:,:].std(0),color='r')
    pp.xticks(range(n_tau))
    if i_type==0:
        pp.xlabel('time shift')
    pp.ylabel(labels[i_type])
pp.savefig(work_dir+'autocovariance')
pp.close()


# kurtosis (4th order)
kurtosis = np.zeros([n_type,n_rep])
for i_type in range(n_type):
    for i_rep in range(n_rep):
        kurtosis[i_type,i_rep] = stt.kurtosis(all_ts[i_type,i_rep,:])
        
pp.figure(figsize=[6,4])
pp.axes([0.2,0.2,0.7,0.7])
pp.violinplot(kurtosis.T,positions=np.arange(n_type)+1)
pp.xticks([1,2,3,4],labels)
pp.axis(xmin=0.5,xmax=n_type+0.5,ymin=-1,ymax=kurtosis.max()+1)
pp.ylabel('kurtosis')
pp.savefig(work_dir+'kurtosis')
pp.close()


# power spectrum
freq = np.fft.fftfreq(T,1)[:int(T/2)]
pws = np.abs(np.fft.fft(all_ts,axis=2)[:,:,:int(T/2)])**2

pp.figure(figsize=[6,4])
pp.axes([0.2,0.2,0.7,0.7])
for i_type in range(n_type):
    pp.errorbar(freq,pws[i_type,:,:].mean(0),yerr=pws[i_type,:,:].std(0)/np.sqrt(n_rep)) # standard error of the meaan
pp.legend(labels)
pp.axis(xmin=0,xmax=0.2)
pp.xlabel('frequencies')
pp.ylabel('power spectrum')
pp.savefig(work_dir+'power_spectrum')
pp.close()


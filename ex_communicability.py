#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:41:57 2018

@author: gilsonmatthieu

This script calculates the (dynamic) communicability for a small network of 4 nodes, 
reproducing parts of Fig. 2 and 3 in the paper:

Gilson M, Kouvaris NE, Deco G, Zamora-López G (2018)
Novel framework to analyze complex network dynamics.
Phys Rev E doi: 10.1103/PhysRevE.97.052301

For the dynamic network (multivariate Ornstein-Uhlenbeck here), dynamic communicability 
quantifies the interactions between nodes over time (via the Green function). Three points 
to see:
    - communicability captures the global network feedback (e.g., open versus close loop)
    - nodes without direct connections end up with non-zero interactions because of network effects
    - the communicability matrix is first aligned with weights, then becomes homogeneous

"""


import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as pp
import os


os.system('clear')

res_dir = 'ex_dyn_comm/'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)


############
# simulation

# network properties
N = 4 # number of nodes
tau_x = 2. # time constant (for exaponential decay)

# (effective) connectivity
c0 = 0.4 # scaling factor
C = np.zeros([N,N])
C[1,0] = 1.2*c0
C[2,1] = 1.1*c0
C[3,2] = 0.7*c0
C[0,3] = 1.*c0

# Jacobian of the dynamic system
J0 = -np.eye(N)/tau_x # diagonal elements of Jacobian
J = J0 + C # full Jacobian with connectivity
renorm_factor = -np.sum(1./J0.diagonal()) # integral value of the diagonal Jacobian

# input variances
Sigma = np.eye(N)

# simulation properties
T = 40. # duration
dt = 1. # time step
vT = np.arange(0,T+dt*0.5,dt) # discrete simulation steps
nT = vT.size

# record values
dyn_comm = np.zeros([nT,N,N]) # dynamic communicability matrix
for iT in range(nT):
    dyn_comm[iT,:,:] = (spl.expm(J*vT[iT]) - spl.expm(J0*vT[iT])) / renorm_factor


#######
# plots

# Jacobian
pp.figure(figsize=[3,3])
pp.axes([0.2,0.25,0.65,0.65])
pp.imshow(J,origin='lower',interpolation='nearest',vmin=-0.5,vmax=0.5,cmap='bwr')
pp.xticks(range(N),np.arange(N)+1,fontsize=12)
pp.yticks(range(N),np.arange(N)+1,fontsize=12)
pp.colorbar()
pp.xlabel('source node',fontsize=12)
pp.ylabel('target node',fontsize=12)
pp.savefig(res_dir+'Jacobian')
pp.close()

# communicability matrix at 3 times
vT2 = [1,4,10]
for iT in vT2:
    pp.figure(figsize=[3,3])
    pp.axes([0.2,0.25,0.65,0.65])
    pp.imshow(dyn_comm[iT,:,:],origin='lower',interpolation='nearest',vmin=0,vmax=0.2/renorm_factor,cmap='Reds')
    pp.xticks(range(N),np.arange(N)+1,fontsize=12)
    pp.yticks(range(N),np.arange(N)+1,fontsize=12)
    pp.colorbar()
    pp.xlabel('source node',fontsize=12)
    pp.ylabel('target node',fontsize=12)
    pp.savefig(res_dir+'ex_dyn_comm_t'+str(iT))
    pp.close()

# total communicability over time
pp.figure(figsize=[4,3])
pp.axes([0.2,0.6,0.75,0.35])
pp.plot(vT,dyn_comm.sum(axis=(1,2)),color='r')
pp.xticks(np.arange(0,T,10),fontsize=8)
pp.yticks([0,0.5],fontsize=8)
pp.axis(xmin=0,xmax=T)
pp.ylabel('total\n communicability',fontsize=8)
pp.axes([0.2,0.15,0.75,0.35])
pp.plot(vT[1:],dyn_comm[1:,:,:].std(axis=(1,2))/dyn_comm[1:,:,:].mean(axis=(1,2)),color='r')
pp.xticks(np.arange(0,T,10),fontsize=8)
pp.yticks([0,1],fontsize=8)
pp.axis(xmin=0,xmax=T)
pp.xlabel('time',fontsize=8)
pp.ylabel('communicability\n diversity',fontsize=8)
pp.savefig(res_dir+'tot_div_comm')
pp.close()

# evolution over time of sum of incoming/outgoing EC-based dynamic communicability for all nodes
T_aff = int(nT/2+1)

pp.figure(figsize=[3,4])
pp.axes([0.17,0.55,0.79,0.4])
pp.imshow(dyn_comm[:T_aff,:,:].sum(2).T,origin='lower',interpolation='nearest',vmin=0,vmax=0.1,aspect=1.5,cmap='jet')
pp.xticks([0,10,20],fontsize=8)
pp.yticks(range(N),np.arange(N)+1,fontsize=8)
pp.ylabel('node',fontsize=8)
cb = pp.colorbar(shrink=0.6,ticks=[0,0.1])
cb.ax.tick_params(labelsize=8)
pp.title('input\ncommunicability',fontsize=9)
pp.axes([0.17,0.08,0.79,0.4])
pp.imshow(dyn_comm[:T_aff,:,:].sum(1).T,origin='lower',interpolation='nearest',vmin=0,vmax=0.1,aspect=1.5,cmap='jet')
pp.xticks([0,10,20],fontsize=8)
pp.yticks(range(N),np.arange(N)+1,fontsize=8)
pp.xlabel('time',fontsize=8)
pp.ylabel('node',fontsize=8)
cb = pp.colorbar(shrink=0.6,ticks=[0,0.1])
cb.ax.tick_params(labelsize=8)
pp.title('output\ncommunicability',fontsize=9)
pp.savefig(res_dir+'t_sum_in_out_comm_J')
pp.close()


# comparison between communicability and EC matrices over time
pp.figure(figsize=[6,3])
for iT in range(len(vT2)): # same for later times t = 5, 10 and 50 TR (dynamic communicability)
    pp.axes([0.14+0.3*iT,0.24,0.25,0.65])
    pp.plot(C,(spl.expm(J*vT2[iT])-spl.expm(J0*iT*dt))/renorm_factor,'.',c='r')
    if iT==0:
        pp.yticks(np.arange(0,0.05,0.02),fontsize=8)
    else:
        pp.yticks(np.arange(0,0.05,0.02),[],fontsize=8)
    pp.xticks([0,0.2,0.4,0.6],fontsize=8)
    pp.axis(xmin=-0.05,xmax=0.52,ymin=0,ymax=0.045)
    if iT==1:
        pp.xlabel('C weight',fontsize=8)
    if iT==0:
        pp.ylabel('communicability',fontsize=8)
    pp.title('t='+str(vT2[iT]),fontsize=8)
pp.savefig(res_dir+'match_EC_comm')
pp.close()



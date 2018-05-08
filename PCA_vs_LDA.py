#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:04:11 2018

@author: gilsonmatthieu

This script shows how principal component analysis (PCA) and linear discriminant analysis (LDA) describe the data samples. Here 2 models:
    - 4 groups of samples in 2+ dimensions differ by their means along the 1st dimension (the variance is common to all groups, but is dimension-specific);
    - 4 groups of samples in 3 dimensions forming a tetrahedron.
Check the performance (as measured by the silhouette values) of the unsupervised PCA and the supervised LDA depending on the variance scaling (e.g., swap the values for the scaling factors). Try also with more dimensions, etc.
This script was inspired by the scikit-learn documentation (http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html).
"""

import numpy as np
import sklearn.discriminant_analysis as sda
import sklearn.metrics as sm
import scipy.spatial.distance as ssd
import matplotlib.pyplot as pp


# sample properties
d = 3 # number of dimensions
c = 4 # number of classes
n = 30 # number of samples per class

# properties of groups: play with variance scaling
m_gp = np.zeros([c,d]) # means
std_gp = np.ones([c,d]) # standard deviation
if False: # configuration 1 with d = 2
    m_gp[:,0] = np.linspace(0,1,c) # difference in mean between groups along 1st dimension
    std_gp[:,0] *= 0.2 # along 1st dimension
    std_gp[:,1:] *= 1.5 # along 1st dimension
else: # configuration 2 with d = 3
    m_gp[:,0] = np.array([0,0,1,1])
    m_gp[:,1] = np.array([0,1,1,0])
    m_gp[:,2] = np.array([0,1,0,1])
    std_gp[:,1:] *= 0.3 # along 2nd dimension

# samples in d dimensions and labels
x = np.zeros([n*c,d])
x_label = np.zeros([n*c])
for i_c in range(c):
    for i_n in range(n):
        x[i_c*n+i_n,:] = m_gp[i_c,:] + np.random.randn(d) * std_gp[i_c,:]
    x_label[i_c*n:(i_c+1)*n] = i_c


###############################
# principal component analysis: first two components
D,U = np.linalg.eig(np.corrcoef(x,rowvar=False)) # D is vector of eigenvalues, U is passage matrix with columns = eigenvectors
ord_eig = np.argsort(np.real(D))[::-1] # sort eigenvalues from largest positive to smallest negative
D = D[ord_eig]
U = U[:,ord_eig]

# new coordinates along 2 first components
y_pca = np.real(np.dot(U.T,x.T))[:2,:].T


##############################
# linear discriminant analysis
lda = sda.LinearDiscriminantAnalysis(n_components=2)
y_lda = lda.fit(x, x_label).transform(x)


####################
# 2D projection plot
cols = ['r','g','b','m','y','c','k'] * int(np.ceil(c/7))

pp.figure(figsize=[6,2])
# original space (first 2 dimensions)
pp.axes([0.12,0.2,0.22,0.7])
for i_c in range(c):
    pp.plot(x[i_c*n:(i_c+1)*n,0],x[i_c*n:(i_c+1)*n,-1],'.',c=cols[i_c])
pp.xticks(fontsize=8)
pp.yticks(fontsize=8)
pp.xlabel('x1',fontsize=8)
pp.ylabel('x2',fontsize=8)
pp.title('original space',fontsize=8)
# PCA space (1st and 2nd principal components)
pp.axes([0.44,0.2,0.22,0.7])
for i_c in range(c):
    pp.plot(y_pca[i_c*n:(i_c+1)*n,0],y_pca[i_c*n:(i_c+1)*n,1],'.',c=cols[i_c])
pp.xticks(fontsize=8)
pp.yticks(fontsize=8)
pp.xlabel('PC1',fontsize=8)
pp.ylabel('PC2',fontsize=8)
pp.title('PCA space',fontsize=8)
# LDA space
pp.axes([0.76,0.2,0.22,0.7])
for i_c in range(c):
    pp.plot(y_lda[i_c*n:(i_c+1)*n,0], y_lda[i_c*n:(i_c+1)*n,1],'.',c=cols[i_c])
pp.xticks(fontsize=8)
pp.yticks(fontsize=8)
pp.xlabel('C1',fontsize=8)
pp.ylabel('C2',fontsize=8)
pp.title('LDA space',fontsize=8)
pp.savefig('2D_projections')
pp.close()


#########################
# silhouette coefficients
sil_coef = np.zeros([c*n,3]) # describes the goodness of clustering of the 2D projection
for ii in range(3):
    # get 
    if ii==0:
        x_tmp = x[:,:2] # first 2 dimensions
    elif ii==1:
        x_tmp = y_pca
    else:
        x_tmp = y_lda
    # calculate silhouette coefficients for each sample with Euclidean distance on 2 dimensions (projection)
    sil_coef[:,ii] = sm.silhouette_samples(x_tmp,x_label,metric=ssd.euclidean)

pp.figure(figsize=[5,3])
for ii in range(3):
    pp.axes([0.1+0.3*ii,0.15,0.25,0.75])
    pp.barh(range(c*n),sil_coef[:,ii])
    pp.axis(xmin=-1,xmax=1)
    pp.xticks([-1,0,1],fontsize=8)
    if ii==0:
        pp.yticks(np.arange(0,c*n+1,n),fontsize=8)
        pp.title('original space',fontsize=8)
    elif ii==1:
        pp.yticks(np.arange(0,c*n+1,n),[],fontsize=8)
        pp.title('PCA space',fontsize=8)
    else:
        pp.yticks(np.arange(0,c*n+1,n),[],fontsize=8)
        pp.title('LDA space',fontsize=8)
pp.savefig('hist_silhouette')
pp.close()



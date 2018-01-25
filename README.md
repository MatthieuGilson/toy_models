# estimate_dir_conn.py

This script is a toy model to estimate (un)directed connectivity from the simulated activity of a multivariate Ornstein-Uhelnbeck (MOU) process. The observed activity is also downsampled before connectivity is estimated from covariance-based methods.

The aim of the script is to illustrate the difference between estimating **undirected** connectivity with partial correlation (based on the zero-lag covariance matrix *Q0*) and **directed** connectivity based on time-shifted covariances of the observed activity (*Q1* in addition to *Q0*). It also shows that the mean activity (*X*) is very similar for all configurations, i.e., hardly informative about the original network parameters.

The script generates activity for 4 configurations of a network of 4 nodes, which differ by the connectivity *C* and the input covariance matrix *Σ* (left of the plotted figure). The observables (mean activity *X* and covariance matrices *Q0* and *Q1*) are in the center of the figure. The network estimates for partial correlation (PC) and the MOU estimation ('est') are located on the right of the figure.

Refs:
en.wikipedia.org/wiki/Partial_correlation
en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

# estimate_dir_conn.py

The aim of this toy model is to illustrate the difference between estimating **undirected** connectivity with partial correlation (based on the zero-lag covariance matrix *Q0*) and **directed** connectivity based on time-shifted covariances of the observed activity (*Q1* in addition to *Q0*). It also shows that the mean activity (*X*) is very similar for all configurations, i.e., hardly informative about the original network parameters.

The script in python3 generates activity using multivariate Ornstein-Uhelnbeck (MOU) process for 4 network configurations. The 4 networks differ by the connectivity *C* and the input covariance matrix *Σ* (left of the plotted figure). The observed activity is also downsampled before calculating the observables: mean activity *X* and covariance matrices *Q0* and *Q1* (center of the figure). The network estimates for partial correlation (PC) and the MOU estimation ('est') are located on the right of the figure. Compare the estimates with the original network parameters.

Refs:
- en.wikipedia.org/wiki/Partial_correlation
- en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process


# MVAR_Granger_detection.py

This script compares the efficiency of conditional Granger causality analysis and a novel non-parametric testing method for MVAR in a network with linear feedback in discrete time (same as MVAR, which is canonical to both estimation methods). It examines the true-detection rates and false-alarm rates for network with random size, density, etc.

Ref:
- Gilson, Tauste Campo, Chen, Thiele, Deco (2017) Net Neurosci doi.org/10.1162/NETN_a_00019

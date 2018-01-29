import sys, os
import numpy as np
import scipy.stats as stt


sim_dir = 'sim_tmp/'
if not os.path.exists(sim_dir):
    print('create directory:', sim_dir)
    os.makedirs(sim_dir)


###########
# network functions

# random matrix with probability of connection p_arg, and max weight w_arg
def gen_random_C(p_arg,min_w_arg,max_w_arg):
    C_tmp = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if np.random.rand()<p_arg:
                C_tmp[i,j] = min_w_arg+(max_w_arg-min_w_arg)*np.random.rand()
    return C_tmp


# run simulation of network activity with noise
def run_sim(T1,T2,C_sim,K_sim):
    t_span = np.arange(T1+T2)
    # initialization
    x_tmp = np.random.rand(N)
    noise = np.random.normal(size=[T1+T2,N],scale=1.)
    # simulation and recording
    x_hist_tmp = np.zeros([T2,N])
    for t in t_span:
        x_tmp = np.dot(C_sim,x_tmp) + np.dot(K_sim,noise[t,:])
        if t>=T1:
            x_hist_tmp[t-T1,:] = x_tmp
    return x_hist_tmp


###############################################################################
# SIMULATION

# simulation duration
T = 3000 # duration of simulation
T0 = 1000 # initialization time for network dynamics

n_shuf = 200 # number of generated surrogates
n_pval = 5 # 5 p-values (1-5%) to test false-alarm rates
sensitivity = 0.02 # p-value for detection
i_sensitivity = int(sensitivity*n_shuf) # equivalent threshold for n_shuf surrogates


# number of network nodes
N = 50+np.random.randint(101)

# connection density
p_C = 0.1+0.2*np.random.rand()

# weight bounds
max_w_C = (0.5+0.4*np.random.rand())/p_C/N
min_w_C = max_w_C*(0.5+0.5*np.random.rand())

# input noise matrix (diagonal on4ly so far)
noise_level = 0.25+0.5*np.random.rand()
spread_noise = np.random.rand()
offdiag_noise_coeff = np.random.rand()/5

C_orig = gen_random_C(p_C,min_w_C,max_w_C)

K_orig = noise_level*(np.eye(N) + offdiag_noise_coeff*gen_random_C(np.sqrt((0.05+0.45*np.random.rand())/N),0,1))
for i in range(N):
    K_orig[i,:] *= (1.-spread_noise/2+spread_noise*np.random.rand()) # modulation of diagonal elements of Sigma
Sigma_orig = np.dot(K_orig,K_orig.T)

# network simulation
X = run_sim(T0,T,C_orig,K_orig)
if X.mean(0).max()>50:
    print('explosion with noise')
    sys.exit()

# masks
mask_diag = np.eye(N,dtype=np.bool)
mask_offdiag = np.logical_not(mask_diag)
mask_C_nonzero = C_orig>0
mask_C_zero = C_orig==0
mask_Sigma_zero = Sigma_orig==0

# center observed time series
t_X = np.array(X)
t_X -= np.outer(np.ones([t_X.shape[0]]),t_X.mean(0))


###############################################################################
# calc MV + shuf
# structures to record results
C_MV = np.zeros([N,N]) # matrix of estimated MVAR coefficients
FP_C_MV = np.zeros([n_pval,2]) # false-alarm rates for several p-values and 2 tests (local and global thresholds)
detect_C_MV = np.zeros([2,N,N],dtype=np.bool) # matrices of detected connections (same as before: local, global)

# covariance matrices
Q0 = np.tensordot(t_X[:-1,:],t_X[:-1,:],axes=[0,0])/float(t_X.shape[0]-2)
Q1 = np.tensordot(t_X[:-1,:],t_X[1:,:],axes=[0,0])/float(t_X.shape[0]-2)
# estimated MVAR coefficients
C_MV = np.dot(np.linalg.pinv(Q0),Q1).T

# generation of surrogates
C_MV_shuf = np.zeros([n_shuf,N,N])

for i_shuf in range(n_shuf):
    # copy original time series for manipulation
    t_X_shuf = np.zeros([T,N])
    for i in range(N):
        t_X_shuf[:,i] = t_X[np.argsort(np.random.rand(T)),i] # time permutation
#        t_X_shuf[:,i] = np.roll(t_X[:,i],np.random.randint(T)) # time rolling

    Q0_shuf_tmp = np.tensordot(t_X_shuf[:-1,:],t_X_shuf[:-1,:],axes=[0,0])/float(t_X.shape[0]-2)
    Q1_shuf_tmp = np.tensordot(t_X_shuf[:-1,:],t_X_shuf[1:,:],axes=[0,0])/float(t_X.shape[0]-2)
    C_MV_shuf[i_shuf,:,:] = np.dot(np.linalg.pinv(Q0_shuf_tmp),Q1_shuf_tmp).T

max_C_MV_ij = np.sort(C_MV_shuf,axis=0) # sort surrogate values for each matrix element
max_C_MV_global = np.sort(C_MV_shuf,axis=None) # sort surrogate values pooling all elements

# calculate false alarm rate
for i_pval in range(n_pval):
    FP_C_MV[i_pval,0] = np.sum(C_MV[mask_C_zero]>=max_C_MV_ij[-int((1+i_pval)/100.*n_shuf),mask_C_zero])/float(mask_C_zero.sum())
    FP_C_MV[i_pval,1] = np.sum(C_MV[mask_C_zero]>=max_C_MV_global[-int((1+i_pval)/100.*n_shuf*(N**2))])/float(mask_C_zero.sum())

# significance test for estimated MV
detect_C_MV[0,:,:] = C_MV>=max_C_MV_ij[-i_sensitivity,:,:]
detect_C_MV[1,:,:] = C_MV>=max_C_MV_global[-int(i_sensitivity*(N**2))]


###############################################################################
# conditional Granger causality analysis
C_GRc = np.zeros([N,N]) # matrix of log ratio
FP_C_GRc_param = np.zeros([n_pval]) # false-alarm rates for several p-values
detect_C_GRc_param = np.zeros([N,N],dtype=np.bool) # matrices of detected connections

# estimate ratios
err_full = np.linalg.lstsq(t_X[:-1,:],t_X[1:,:],rcond=None)[1]                
for j in range(N):
    v_bool = np.ones([N],dtype=np.bool)
    v_bool[j] = False
    err_noj = np.linalg.lstsq(t_X[:-1,v_bool],t_X[1:,:],rcond=None)[1]
    C_GRc[:,j] = np.log(err_noj/err_full)

# calculate false alarm rate
for i_pval in range(n_pval):
    lim_GR = stt.f.isf((i_pval+1)/100.,1,T-N-1)/(T-N-1) # threshold for parametric test
    FP_C_GRc_param[i_pval] = np.sum(C_GRc[mask_C_zero]>=lim_GR)/float(mask_C_zero.sum())

# significance test for Granger causality analysis
lim_GR = stt.f.isf(sensitivity,1,T-N-1)/(T-N-1) # threshold for parametric test 
detect_C_GRc_param = np.exp(C_GRc)-1>=lim_GR        


###############################################################################
# save data
np.save(sim_dir+'C_orig.npy',C_orig)

np.save(sim_dir+'C_MV.npy',C_MV)
np.save(sim_dir+'C_GRc.npy',C_GRc)

np.save(sim_dir+'FP_C_MV.npy',FP_C_MV)
np.save(sim_dir+'detect_C_MV.npy',detect_C_MV)

np.save(sim_dir+'FP_C_GRc_param.npy',FP_C_GRc_param)
np.save(sim_dir+'detect_C_GRc_param.npy',detect_C_GRc_param)


# display results
print('network properties: {0:d} nodes with {1:d}% density; min and max weights {2:.2f}-{3:.2f}; input variances {4:.2f}.'.format(N, int(100*p_C), min_w_C, max_w_C, noise_level))
print()

print('non-parametric MVAR + local test:')
i_type_ref = 0
print('false-alarm rate (expected 1,2,3,4,5%)', FP_C_MV[:,i_type_ref])
print('true positive (for {0:d}% expected false alarms):'.format(int(100*sensitivity)), np.logical_and(detect_C_MV[i_type_ref,:,:],C_orig>0).sum(), 'for', (C_orig>0).sum(), 'existing connections')
print()
i_type_ref = 1
print('non-parametric MVAR + global test:')
print('false-alarm rate (expected 1,2,3,4,5%)', FP_C_MV[:,i_type_ref])
print('true positive (for {0:d}% expected false alarms):'.format(int(100*sensitivity)), np.logical_and(detect_C_MV[i_type_ref,:,:],C_orig>0).sum(), 'for', (C_orig>0).sum(), 'existing connections')
print()
print('parametric Granger causality:')
print('false-alarm rate (expected 1,2,3,4,5%)', FP_C_GRc_param)
print('true positive (for {0:d}% expected false alarms):'.format(int(100*sensitivity)), np.logical_and(detect_C_GRc_param,C_orig>0).sum(), 'for', (C_orig>0).sum(), 'existing connections')

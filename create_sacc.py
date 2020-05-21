import os
import sacc
import sys
import numpy as np
import pyccl as ccl
import copy
import shutil
import matplotlib.pyplot as plt

from desclss.halo_mod_corr import HaloModCorrection
from calculate_smooth_s_and_prior import get_smooth_s_and_prior
from HSC_wrapper import obtain_Tmat
from HSC_wrapper import get_theory

# Use realistic NG covariance
# read = 'NEWCOV_COADDED'; write = 'NEWCOV_MARG'
# Important REMOVE l-cuts for the real run
# TODO we are assuming cosmic variance of the tomographic bins is uncorrelated

try:
    # Names of the data read and write directories 
    read = sys.argv[1]
    write = sys.argv[2]
except:
    print("Usage of this script is: python create_sacc.py <read_dir> <write_dir>")
dir_read = "data/"+read
dir_write = "data/"+write

# l-cuts
# Remove this for the actual MCMC run as the sampler makes its own cut TODO
lmax = [2170.58958919, 2515.39193451, 3185.36076391, 4017.39370804]

# choice for noise and smoothing

#noi_fac, A_smooth, pert = 4., 0.15, 'cov'
#noi_fac, A_smooth, pert = 42., 0.1, 'cov'
#noi_fac, A_smooth, pert = 4000., 0.03, 'cov'
#noi_fac, A_smooth, pert = 42., 0.1, 'shiftone'
#noi_fac, A_smooth, pert = 42., 0.0, 'shiftone'
noi_fac, A_smooth, pert = 420., 0.0, 'shiftall'


# Cosmological parameters
# Checked cosmo_params are the same as what the MCMC chain fixes for them
cosmo_params = {'n_s': 0.9649,
                'sigma8':0.8111,
                'h':0.6736,
                'Omega_c':0.264,
                'Omega_b':0.0493}

# HOD params: assuming posterior mean from paper
hod_params = {'zfid':0.65,
              'lmmin':11.88, 'lmminp':-0.5,##
              'sigm_0':0.4, 'sigm_1':0,
              'm0':5.7, 'm0p':2.5,##
              'm1':13.08, 'm1p':0.9,##
              'alpha_0':1, 'alpha_1':0,
              'fc_0':1., 'fc_1':0.}

hod_sigmas = {'lmmin':0.22, 'lmminp':2.0,
              'm0':4.0, 'm0p':5.0,
              'm1':0.27, 'm1p':2.6}

Z_params = {'zshift_bin0':0.000,'zshift_bin1':-0.016,\
            'zshift_bin2':-0.01,'zshift_bin3':0.01,\
            'zwidth_bin0':-0.05,'zwidth_bin1':-0.01,\
            'zwidth_bin2':0.035,'zwidth_bin3':0.05}

Z_params0 = {'zshift_bin0':0.000,'zshift_bin1':-0.0,\
            'zshift_bin2':-0.0,'zshift_bin3':0.0,\
            'zwidth_bin0':-0.0,'zwidth_bin1':-0.0,\
            'zwidth_bin2':0.0,'zwidth_bin3':0.0}

# Here are some choice that we make for the tests we are running
# TEST #1
if "1sigma" in write:
    # Delete the CV covmat and the Tmat
    print("Getting marginalized covariance for the 1-sigma test")
    coin = 1
    for key in hod_sigmas.keys():
        hod_params[key] += coin*hod_sigmas[key]
        coin *= -1
        print(key,hod_params[key])

# TEST #2
if "nosmooth" in write:
    # Delete the CV covmat
    print("Getting marginalized covariance for the no-smoothing test")
    A_smooth = 0.

# TEST #3s
if "largenoise" in write:
    # Delete the CV covmat
    print("Getting marginalized covariance for the large-noise test")
    noi_fac = 42.

# TEST #4 there is this test too "MARG_nosmooth_largenoise":


# This is the improved covariance with David's simplification (Woodbury matrix identity)
# NB: the T matrix here is the transpose of what is written in the equations on overleaf
def obtain_improved_cov(cov,Tmat,prior):
    # The expressions inside the bracket, before the bracket and after the bracket
    cov_m = cov + np.dot(np.dot(Tmat.T,np.linalg.inv(prior)),Tmat)
    return cov_m

# Auxiliary function to get the Nzs for all (4) tracers
def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])


# Theory prediction
cosmo = ccl.Cosmology(**cosmo_params)

# Halo Model Correction: have checked that this is the same as what is applied for the MCMC run
HMCorrection = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

# Load the data whose precision matrix we would like to modify
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")
s_noi = sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")

# redshift array
zar = s_d.tracers[0].z


# Calculate the smooth s_m = (P0+D)^-1 P0 s0 and smooth prior prior_smo = P0+D
s_m, prior_smo = get_smooth_s_and_prior(s_d,cosmo,want_prior=True,A_smooth=A_smooth,noi_fac=noi_fac)

# Number of tracers and z-bins
Nz_per_tracer = len(s_d.tracers[0].z)
N_tracers = len(s_d.tracers)
Nz_total = N_tracers*Nz_per_tracer

# Applying l-cuts; remove for MCMC run

# Get the covariance and precision matrices
N_data = len(s_d.mean.vector)
cov = s_d.precision.getCovarianceMatrix()
prec = np.linalg.inv(cov)

# Tmat computation
if os.path.isfile("Tmat_"+read+".npy"):
    print ("Loading cached Tmat")
    Tmat = np.load("Tmat_"+read+".npy")
else:
    Tmat = obtain_Tmat(s_d,s_noi,hod_params,Z_params,HMCorrection)
    np.save("Tmat_"+read+".npy",Tmat)

# Obtain the new covariance matrix
cov_CVnoismo = obtain_improved_cov(cov,Tmat,prior_smo)

# Weirdly it seems like this object actually expects the covariance rather than the precision 
cov_n = sacc.Precision(cov_CVnoismo)

# Create new sacc file (TODO: ask about "dense")
s_n = sacc.SACC(s_d.tracers,s_d.binning,s_d.mean.vector,cov_n,meta=s_d.meta)


# Save the new covariance and the power spectrum of the data which shall be run via MCMC
if not os.path.exists(dir_write):
    os.makedirs(dir_write)
s_n.saveToHDF(dir_write+"/power_spectra_wdpj.sacc")

# Copying noise bias file
shutil.copy(dir_read +"/noi_bias.sacc",dir_write)

# !!!!! Everything below is just for checking that the code works !!!!!!


# Sanity check of implementation
prec_CVnoismo = s_n.precision.getPrecisionMatrix()

# Degrees of freedom
dof = N_data

# Function to calculate chi2
def get_chi2(di,precision):
    # Report the chi2
    return np.dot(di,np.dot(precision,di))

## there are already culled in ell

prec_o = s_d.precision.getPrecisionMatrix()
prec_n = s_n.precision.getPrecisionMatrix()
# Computing Cls directly from the data

cl_theory = get_theory(s_d,s_noi,hod_params,Z_params,HMCorrection)
cl_theory_s = get_theory(s_m,s_noi,hod_params,Z_params,HMCorrection)




# xar = np.arange(len(cl_theory))
# errv = np.sqrt(s_d.precision.getCovarianceMatrix().diagonal())
# #plt.plot(xar,cl_theory,'b-')
# plt.errorbar(xar,(s_d.mean.vector-cl_theory)/errv)

# plt.show()


di = s_d.mean.vector - cl_theory
chi2o = get_chi2(di,prec_o)
chi2n = get_chi2(di,prec_n)
print ("Chi2 original model, original cov:", chi2o, dof)
print ("Chi2 original model, new cov ", chi2n)


di = s_d.mean.vector - cl_theory_s
chi2o = get_chi2(di,prec_o)
chi2n = get_chi2(di,prec_n)
print ("Chi2 original model, smoothed Nz, original cov:", chi2o, dof)
print ("Chi2 original model, smoothed Nz, new cov ", chi2n)




#Lets perturb dNz, draw from prior_cov
prior_cov = np.linalg.inv(prior_smo)
Nz = NzVec(s_d)
Nz_s = NzVec(s_m)
dNz = np.random.multivariate_normal(np.zeros_like(Nz),prior_cov)
print ("Sanity: ",np.dot(dNz,np.dot(np.linalg.inv(prior_cov),dNz)))
#plt.errorbar(np.arange(len(Nz_s)),Nz_s, yerr=np.sqrt(prior_cov.diagonal()))
#plt.show()
#stop()


if pert == 'cov':
    NzP = Nz+dNz
elif pert == 'shiftone':
    ## let's instead implement a shift by one bin by 0.02
    NzP = np.copy(Nz)
    NzP [200:299] = Nz[201:300]
    dNz = NzP-Nz
elif pert == 'shiftall':
    NzP = np.copy(Nz)
    for i in range (4):
        NzP [i*100:i*100+99] = Nz[i*100+1:i*100+100]
        NzP [i*100+99]=0.0
    dNz = NzP-Nz
else:
    raise NotImplementedError

    
    
print ("Sanity: ",np.dot(dNz,np.dot(np.linalg.inv(prior_cov),dNz)))




if True:
    fig, ax = plt.subplots(4,1, facecolor="w",
            gridspec_kw={"hspace": 0.0},
            figsize=(10, 6))
    for i in range(4):
        li = i*100
        hi = li+100
        ax[i].plot(zar,Nz[li:hi],'r-')
        ax[i].plot(zar,Nz_s[li:hi],'b-')
        ax[i].plot(zar,NzP[li:hi],'g-')
        ax[i].set_xlim(0,2.6)
        ax[i].set_ylabel('$N_%i (z)$'%(i+1))
    ax[3].set_xlabel('z')
    ax[0].plot([],[],'r-',label='original')
    ax[0].plot([],[],'b-',label='smoothed')
    ax[0].plot([],[],'g-',label='shifted')
    ax[0].legend(ncol=3)
    plt.savefig('fig1.png')
        
    plt.show()
    

s_dp = copy.copy(s_d)
i=0
for t in s_dp.tracers:
    #print (t.z)
    tl = len(t.Nz)
    t.Nz = NzP[i:i+tl]
    i+=tl

# Slightly improved computation using taylor expansion
cl_theory_perturbed = get_theory(s_dp,s_noi,hod_params,Z_params,HMCorrection)
cl_theory_taylor = cl_theory + np.dot(Tmat.T,dNz)

di = s_d.mean.vector - cl_theory_perturbed
chi2po = get_chi2(di,prec_o)
chi2pn = get_chi2(di,prec_n)
print ("Chi2 perturbed model, original cov:", chi2po, dof)
print ("Chi2 perturbed model, new cov ", chi2pn)

di = s_d.mean.vector - cl_theory_taylor
chi2to = get_chi2(di,prec_o)
chi2tn = get_chi2(di,prec_n)
print ("Chi2 taylor model, original cov:", chi2to, dof)
print ("Chi2 taylor model, new cov ", chi2tn)



# Log determinant ratio
log_det_prec = np.linalg.slogdet(prec)[1]
log_det_prec_CVnoismo = np.linalg.slogdet(prec_CVnoismo)[1]
ratio_det_CVnoismo = log_det_prec-log_det_prec_CVnoismo
ratio_det_CVnoismo = np.exp(ratio_det_CVnoismo)
print("Ratio of determinants of original precision to CV+noise+smooth precision per dof = ", ratio_det_CVnoismo**(1./dof))
# for the test data, the answer used to be ~1.12 and is now ~1.3 with COADD

if True:
    fig, ax = plt.subplots(4,4, facecolor="w",
            gridspec_kw={"hspace": 0.0, "wspace":0},
            figsize=(10, 10))
    data = s_d.mean.vector
    err = np.sqrt(cov.diagonal())
    err2 = np.sqrt(s_n.precision.getCovarianceMatrix().diagonal())
    for i in range(4):
        for j in range (i,4):
            cax = ax[j,i]
            ndx = s_d.ilrange(i,j)
            ell = s_d.binning.binar['ls'][ndx]
            well = ell**1.2
            cax.plot(ell,well*cl_theory[ndx],'r-')
            cax.plot(ell,well*cl_theory_s[ndx],'g-')
            cax.plot(ell,well*cl_theory_perturbed[ndx],'b-')
            cax.plot(ell,well*cl_theory_taylor[ndx],'c:')
            cax.errorbar(ell,data[ndx]*well,yerr=err[ndx]*well,fmt='ko-')
            #cax.errorbar(ell,data[ndx]*well,yerr=err2[ndx]*well,fmt='ko:')
    ax[0,3].plot([],[],'r-',label='default theory')
    ax[0,3].plot([],[],'g-',label='smooth theory')
    ax[0,3].plot([],[],'b-',label='perturb theory')
    ax[0,3].plot([],[],'c:',label='taylor theory')
    ax[0,3].legend(frameon=False)
    plt.savefig('fig2.png')
    plt.show()


    print (s_d.binning.binar['T1'])
    #plt.plot(cl_theory/cl_theory)
    #plt.plot(cl_theory_s/cl_theory)
    #plt.plot(cl_theory_perturbed/cl_theory)
    #plt.plot(cl_theory_taylor/cl_theory)
    #plt.show()



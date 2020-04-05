import os
import sacc
import sys
import numpy as np
import pyccl as ccl
import copy
from modules.halo_mod_corr import HaloModCorrection
from modules.theory_cls import get_theory
from compute_CV_cov import compute_covmat_cv
from smoothness import obtain_smoothing_D

# Note to self and reader -> this should eventually be run with the real data, i.e.
# read = 'COADD'
# For now I am using read = 'test' and write = 'test_2' 

try:
    # Names of the data read and write directories 
    read = sys.argv[1]
    write = sys.argv[2]
except:
    print("Usage of this script is: python create_sacc.py <read_dir> <write_dir>")
dir_read = "data/"+read
dir_write = "data/"+write

# l-cuts
# TODO: do we want this? I think not for the real data cause MCMC does this itself
lmax = [2000,2000,2600,3200]

# Cosmological parameters
# Checked cosmo_params are the same as what the MCMC chain fixes for them
cosmo_params = {'n_s': 0.9649,
                'sigma8':0.8111,
                'h':0.6736,
                'Omega_c':0.264,
                'Omega_b':0.0493}

# HOD params TODO: what do we assume for them? Right now they are posterior mean from paper except for m0, m0p (why?)
hod_params = {'zfid':0.65,
              'lmmin':11.88, 'lmminp':-0.5,##
              'sigm_0':0.4, 'sigm_1':0,
              'm0':9., 'm0p':6.,##
              'm1':13.08, 'm1p':0.9,##
              'alpha_0':1, 'alpha_1':0,
              'fc_0':1., 'fc_1':0.}


# Implementation of the CV+noise precision matrix from Eq. 7 in the HSC Nz marg overleaf
# NB: the T matrix here is the transpose of what is written in the equations there
def obtain_improved_prec(prec,Tmat,prior):
    # The expressions inside the bracket, before the bracket and after the bracket
    bracket = np.linalg.inv((np.dot(np.dot(Tmat,prec),Tmat.T)+prior))
    prebracket = np.dot(prec,Tmat.T)
    postbracket = np.dot(Tmat,prec)
    # Assembling everything into one
    prec_n = prec - np.dot(np.dot(prebracket,bracket),postbracket)
    return prec_n

# The T matrix is simply a matrix containing the first der of Cl with resp to Nz
def obtain_Tmat(s_data,Nz,Nztr,hod_pars,cosmo_pars,HMCorr,delta=1.e-3):
    # Create T matrix
    Tmat = []
    for i in range(Nz):
        print ("%i / %i "%(i,Nz))
        s_tmp = copy.deepcopy(s_data)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] += delta
        cl_plus = get_theory(hod_pars, cosmo_pars, s_tmp, halo_mod_corrector=HMCorr)    
        s_tmp = copy.deepcopy(s_data)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] -= delta
        cl_minus = get_theory(hod_pars, cosmo_pars, s_tmp, halo_mod_corrector=HMCorr)    
        Tmat.append((cl_plus-cl_minus)/(2*delta))
        print (Tmat[-1])
    Tmat = np.array(Tmat)
    return Tmat

# Auxiliary function to get the Nzs for all (4) tracers
def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

# Compute the smoothed prior with CV and noise
def obtain_prior_smo(s_data,s_mean,Nz,Ntr,Nztr,cosmology):
    # TODO: s_mean -> What should we do about it
    dNz = NzVec(s_mean)-NzVec(s_data)

    # assume this is a good proxy for the prior for now
    covmat_noise = np.diag(2.*dNz**2)

    # Estimating cosmic variance
    # total cv covmat
    covmat_cv = np.zeros((Nz,Nz))
    for i in range(Ntr):
        # cosmic variance covmat for each tracer
        covmat_cv_per_tracer = compute_covmat_cv(cosmology,s_mean.tracers[i].z,s_mean.tracers[i].Nz)
        covmat_cv[i*Nztr:(i+1)*Nztr,i*Nztr:(i+1)*Nztr] = covmat_cv_per_tracer
        
    # impose smoothness
    A_smooth = 0.5
    smooth_prior = A_smooth**2*obtain_smoothing_D(s_mean,first=True,second=True)

    # obtain prior with CV and noise
    prior = np.linalg.inv(covmat_cv+covmat_noise)
    # obtain the complete smooth prior
    prior_smo = prior+smooth_prior
    return prior_smo


# Theory prediction
cosmo = ccl.Cosmology(**cosmo_params)

# Halo Model Correction: have checked that this is the same as what is applied for the MCMC run
HMCorrection = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

# Load the data whose precision matrix we would like to modify
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")

# Load the mean which is only used to estimate the prior for now
s_m = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj_mean.sacc")

# Number of tracers and z-bins
Nz_per_tracer = len(s_d.tracers[0].z)
N_tracers = len(s_d.tracers)
Nz_total = N_tracers*Nz_per_tracer

# Get the precision matrix after applying l-cuts
s_d.cullLminLmax([0,0,0,0],lmax)
N_data = len(s_d.mean.vector)
prec = s_d.precision.getPrecisionMatrix()

# Tmat computation
if os.path.isfile("Tmat_"+write+".npy"):
    print ("Loading cached Tmat")
    Tmat = np.load("Tmat_"+write+".npy")
else:
    Tmat = obtain_Tmat(s_d,Nz_total,Nz_per_tracer,hod_params,cosmo_params,HMCorrection)
    np.save("Tmat_"+write+".npy",Tmat)

# prior_smo computation
if os.path.isfile("prior_smo_"+write+".npy"):
    print ("Loading cached prior with CV, noise and smoothing")
    prior_smo = np.load("prior_smo_"+write+".npy")
else:
    prior_smo = obtain_prior_smo(s_d,s_m,Nz_total,N_tracers,Nz_per_tracer,cosmo)
    np.save("prior_smo_"+write+".npy",prior_smo)

# Obtain the new precision matrix
prec_CVnoismo = obtain_improved_prec(prec,Tmat,prior_smo)

# Weirdly it seems like this object actually expects the covariance rather than the precision 
prec_n = sacc.Precision(np.linalg.inv(prec_CVnoismo))

# Create new sacc file
s_n = sacc.SACC(s_d.tracers,s_d.binning,s_d.mean.vector,prec_n,meta=s_d.meta)

# Sanity check of implementation
prec_CVnoismo = s_n.precision.getPrecisionMatrix()

# Degrees of freedom
dof = N_data

# log determinant ratio
log_det_prec = np.linalg.slogdet(prec)[1]
log_det_prec_CVnoismo = np.linalg.slogdet(prec_CVnoismo)[1]
ratio_det_CVnoismo = log_det_prec-log_det_prec_CVnoismo
ratio_det_CVnoismo = np.exp(ratio_det_CVnoismo)
print("ratio of det of old prec to CV+noise+smooth prec per dof = ", ratio_det_CVnoismo**(1./dof))
# for the test data, the answer should be 1.1262623661031577 on B.H.'s computer, which it is
# after change a bunch of pyccl versions it's slightly different, but I think that's fine
# 1.1197273972281012

# TODO: what about the noise bias? (perhaps same as noi_bias in read directory), so copy noise bias file?
# Save the new covariance and the power spectrum of the data which shall be run via MCMC
if not os.path.exists(dir_write):
    os.makedirs(dir_write)
s_n.saveToHDF(dir_write+"/power_spectra_wdpj.sacc")

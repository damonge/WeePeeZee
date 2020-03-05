import sys
import os
import sacc
import numpy as np
import pyccl as ccl
import copy
from modules.theory_cls import get_theory
from modules.halo_mod_corr import HaloModCorrection
import matplotlib.pyplot as plt

plot_Nz=False
lmax=[2000,2000,2600,3200]


sim_name = sys.argv[1]
dirname = "data/" + sim_name
# Parameters
cosmo_params = {'n_s': 0.9649,
                'sigma8':0.8111,
                'h':0.6736,
                'Omega_c':0.264,
                'Omega_b':0.0493}
hod_params = {'zfid':0.65,
              'lmmin':11.88, 'lmminp':-0.5,
              'sigm_0':0.4, 'sigm_1':0,
              'm0':9., 'm0p':6.,
              'm1':13.08, 'm1p':0.9,
              'alpha_0':1, 'alpha_1':0,
              'fc_0':1., 'fc_1':0.}


## Get true sacc for true N(z)
s_mean=sacc.SACC.loadFromHDF(dirname+"/power_spectra_wdpj_mean.sacc")
s_data=sacc.SACC.loadFromHDF(dirname+"/power_spectra_wdpj.sacc")

# get lmin, lmax
s_mean.cullLminLmax([0,0,0,0],lmax)
s_data.cullLminLmax([0,0,0,0],lmax)

# Theory prediction
cosmo = ccl.Cosmology(n_s=0.9649, sigma8=0.8111, h=0.6736, Omega_c=0.264, Omega_b=0.0493)
HMCorr = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)
cl_theory_true = get_theory(hod_params, cosmo_params, s_mean, halo_mod_corrector=HMCorr)
cl_theory_naive = get_theory(hod_params, cosmo_params, s_data, halo_mod_corrector=HMCorr)
prec=s_data.precision.getPrecisionMatrix()
di_true = s_data.mean.vector - cl_theory_true
di_naive = s_data.mean.vector - cl_theory_naive
print ("dof = ", len(di_true))
print ("Chi2 wrt to truth = ",np.dot(di_true,np.dot(prec,di_true)))
print ("Chi2 wrt to naive = ",np.dot(di_naive,np.dot(prec,di_naive)))


if plot_Nz:
    plt.figure()
    for i, (trtrue, trnaive) in enumerate(zip(s_mean.tracers, s_data.tracers)):
        za=trtrue.z
        print (trnaive.Nz)
        nz_true = trtrue.Nz / trtrue.Nz.sum()
        nz_naive = trnaive.Nz / trnaive.Nz.sum()
        plt.subplot(2,2,i+1)
        plt.plot(za,nz_true,label='true')
        plt.plot(za,nz_naive,label='naive')
        plt.legend()
    plt.savefig('Nz.pdf')
    plt.show()

## Let's calculate derivative matrix.
Ntr=len(s_mean.tracers) # num tracers
Nztr=len(s_mean.tracers[0].z) # zbins per tracer
Nz=Ntr*Nztr ## total Nz points
Nd=len(s_data.mean.vector) ## total data points


if os.path.isfile("Tmat.npy"):
    print ("Loading cached Tmat")
    Tmat=np.load("Tmat.npy")
else:
    # Make T matrix
    delta=1e-3
    Tmat=[]
    for i in range(Nz):
        print ("%i / %i "%(i,Nz))
        s_tmp=copy.deepcopy(s_data)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] += delta
        cl_plus = get_theory(hod_params, cosmo_params, s_tmp, halo_mod_corrector=HMCorr)    
        s_tmp=copy.deepcopy(s_data)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] -= delta
        cl_minus = get_theory(hod_params, cosmo_params, s_tmp, halo_mod_corrector=HMCorr)    
        Tmat.append((cl_plus-cl_minus)/(2*delta))
        print (Tmat[-1])
    Tmat=np.array(Tmat)
    np.save("Tmat.npy",Tmat)

## let's start with shit theory, preturb it to the rigth Nz and see
## if victory is neight
def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

dNz=NzVec(s_mean)-NzVec(s_data)
cl_theory_taylor=cl_theory_naive+np.dot(Tmat.T,dNz)
di = s_data.mean.vector - cl_theory_taylor
print ("Chi2 wrt to naive + Taylor = ",np.dot(di,np.dot(prec,di)))

# Implementation of the new precision matrix from Eq. 7 in the HSC Nz marg overleaf
# NB: the T matrix here is the transpose of what is written in the equations there
# assume this is a good proxy for the prior for now
prior = np.diag(1./(dNz**2))
# The expressions inside the bracket, before the bracket and after the bracket
bracket = np.linalg.inv((np.dot(np.dot(Tmat,prec),Tmat.T)+prior))
prebracket = np.dot(prec,Tmat.T)
postbracket = np.dot(Tmat,prec)
# Assembling everything into one
prec_new = prec - np.dot(np.dot(prebracket,bracket),postbracket)
# Report the new chi2
print ("Chi2 wrt to naive + Taylor + new precision = ",np.dot(di,np.dot(prec_new,di)))

# Check if positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

print("Prec is pos def? ",is_pos_def(prec))
print("New prec pos def? ",is_pos_def(prec_new))

# Determinant ratio
#prec = np.linalg.inv(prec)
#prec_new = np.linalg.inv(prec_new)

log_det_prec = np.linalg.slogdet(prec)[1]
log_det_prec_new = np.linalg.slogdet(prec_new)[1]
ratio_det = log_det_prec-log_det_prec_new
ratio_det = np.exp(ratio_det)
print("ratio of det of old prec to new prec = ", ratio_det)

# TODO: insert the smoothness; account for the COSMOS bias; run HSC chains anew

import sys
import os
import sacc
import numpy as np
import pyccl as ccl
from modules.theory_cls import get_theory
from modules.halo_mod_corr import HaloModCorrection
import matplotlib.pyplot as plt

plot_Nz=True

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
        nz_true = trtrue.Nz / trtrue.Nz.sum()
        nz_naive = trnaive.Nz / trnaive.Nz.sum()
        plt.subplot(2,2,i+1)
        plt.plot(za,nz_true,label='true')
        plt.plot(za,nz_naive,label='naive')
        plt.legend()
    plt.show()
    
        


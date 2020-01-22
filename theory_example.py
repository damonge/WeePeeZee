import pyccl as ccl
import sacc
import numpy as np
from modules.theory_cls import get_theory
from modules.halo_mod_corr import HaloModCorrection
import matplotlib.pyplot as plt

# Halo model correction
cosmo = ccl.Cosmology(n_s=0.9649, sigma8=0.8111, h=0.6736, Omega_c=0.264, Omega_b=0.0493)
HMCorr = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

# Data
s_d=sacc.SACC.loadFromHDF("data/COADD/power_spectra_wdpj.sacc")
s_n=sacc.SACC.loadFromHDF("data/COADD/noi_bias.sacc")

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

# Theory prediction
cl_theory = get_theory(hod_params, cosmo_params, s_d, halo_mod_corrector=HMCorr)

# A few plots
errs = np.sqrt(np.diag(s_d.precision.getCovarianceMatrix()))
for i1, i2, _, ells_binned, ndx in s_d.sortTracers() :
    plt.figure()
    plt.title('%d %d'%(i1,i2))
    lfac = ells_binned**1.3
    plt.errorbar(ells_binned, lfac * (s_d.mean.vector[ndx] - s_n.mean.vector[ndx]),
                 yerr=lfac * errs[ndx])
    plt.plot(ells_binned, lfac * cl_theory[ndx])
    plt.xlim([0,3000])
plt.show()

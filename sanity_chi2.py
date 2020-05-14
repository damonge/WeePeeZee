#!/usr/bin/env python
import os
import sacc
import sys
import numpy as np
import pyccl as ccl
import copy
import shutil
import matplotlib.pyplot as plt
from modules.halo_mod_corr import HaloModCorrection
from modules.theory_cls import get_theory
from calculate_smooth_s_and_prior import get_smooth_s_and_prior
import scipy.linalg as la


s_d = sacc.SACC.loadFromHDF("data/NEWCOV_COADDED/power_spectra_wdpj.sacc")
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

cosmo = ccl.Cosmology(**cosmo_params)
HMCorrection = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)
cl_theory = get_theory(hod_params,cosmo_params, s_d,halo_mod_corrector=HMCorrection)

di = s_d.mean.vector - cl_theory
prec = s_d.precision.getPrecisionMatrix()
chi2 = np.dot(di,np.dot(prec,di))
print ("chi2 = ",chi2, "dof = ", len(di))

import sys
import os
import sacc
import numpy as np
import pyccl as ccl
from modules.theory_cls import get_theory
from modules.halo_mod_corr import HaloModCorrection


if len(sys.argv) != 4:
    print("Usage: data_generator.py sim_name save_mean (0 or 1) n_svd (number of principal eigenmodes)")
    exit(1)
n_svd = int(sys.argv[3])
save_mean = bool(int(sys.argv[2]))
sim_name = sys.argv[1]

pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz']
s_d = sacc.SACC.loadFromHDF("data/COADD/power_spectra_wdpj.sacc")

def get_new_tracers(add_random=True, n_svd=4):
    tr = []
    for i in range(4):
        zs = s_d.tracers[i].z
        nzs = [s_d.tracers[i].Nz / np.sum(s_d.tracers[i].Nz)]
        for pn in pz_codes:
            n = s_d.tracers[i].extra_cols[pn]
            nzs.append(n / np.sum(n))
        nzs = np.array(nzs)
    
        nz_mean = np.mean(nzs, axis=0)
        nz_new = nz_mean.copy()
        if add_random:
            cov_nzs = np.mean(np.array([(n-nz_mean)[:,None] * (n-nz_mean)[None,:]
                                        for n in nzs]),
                              axis=0)
            w, ev = np.linalg.eigh(cov_nzs)
            sigs = np.sqrt(w[-n_svd:])
            evs = ev[:,-n_svd:]
            nz_new += np.sum((np.random.randn(n_svd) * sigs)[None, :] * evs,
                             axis=1)
        T = sacc.Tracer('bin_%d'%i, 'point',
                        zs, nz_new, exp_sample='HSC_DESC')
        tr.append(T)
    return tr

tr_mean = get_new_tracers(add_random=False)
tr_rand = get_new_tracers(add_random=True, n_svd=4)

# SACC object containing the mean N(z) (which we will treat as the truth)
s_mean = sacc.SACC(tr_mean, s_d.binning, s_d.mean)

# Halo model correction
cosmo = ccl.Cosmology(n_s=0.9649, sigma8=0.8111, h=0.6736, Omega_c=0.264, Omega_b=0.0493)
HMCorr = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

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
cl_theory = get_theory(hod_params, cosmo_params, s_mean, halo_mod_corrector=HMCorr)

# Perturb true spectra
cl_new = np.random.multivariate_normal(cl_theory, s_d.precision.getCovarianceMatrix())

# SACC object with random realization of the N(z)s and the power spectra
s_new = sacc.SACC(tr_rand, s_d.binning, sacc.MeanVec(cl_new),
                  precision=s_d.precision, meta=s_d.meta)

dirname = "data/" + sim_name
os.system("mkdir -p " + dirname)
os.system("cp data/COADD/noi_bias.sacc " + dirname)
s_new.saveToHDF(dirname + "/power_spectra_wdpj.sacc")

if save_mean:
    # SACC object with mean N(z) and mean power spectra
    s_truth = sacc.SACC(tr_mean, s_d.binning, sacc.MeanVec(cl_theory),
                        precision=s_d.precision, meta=s_d.meta)
    s_truth.saveToHDF(dirname + "/power_spectra_wdpj_mean.sacc")

import sacc
import numpy as np
import matplotlib.pyplot as plt
from compute_CV_cov import compute_covmat_cv
from smoothness import obtain_smoothing_D
import pyccl as ccl

pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz']
s_d = sacc.SACC.loadFromHDF("data/COADD/power_spectra_wdpj.sacc")

Nz_per_tracer = len(s_d.tracers[0].z)
N_tracers = len(s_d.tracers)
Nz_total = N_tracers*Nz_per_tracer

cosmo_params = {'n_s': 0.9649,
                'sigma8':0.8111,
                'h':0.6736,
                'Omega_c':0.264,
                'Omega_b':0.0493}

# Theory prediction
cosmo = ccl.Cosmology(**cosmo_params)

def get_mean_cov(s_data,Ntr,Nztr):
    noi_fac = 4.
    cov_all = np.zeros((Nztr*Ntr,Nztr*Ntr))
    
    tr = []
    for i in range(Ntr):
        zs = s_data.tracers[i].z
        nzs = [s_data.tracers[i].Nz / np.sum(s_data.tracers[i].Nz)]
        for pn in pz_codes:
            n = s_data.tracers[i].extra_cols[pn]
            nzs.append(n/np.sum(n))
        nzs = np.array(nzs)
    
        nz_mean = np.mean(nzs, axis=0)
        nz_var = np.var(nzs,axis=0)
        nz_var_mean = nz_var[nz_mean>0].mean()
        cov = np.diag(noi_fac*nz_var_mean*np.ones(len(zs)))
        cov_all[i*len(zs):(i+1)*len(zs),i*len(zs):(i+1)*len(zs)] = cov
        

        T = sacc.Tracer('bin_%d'%i, 'point',
                        zs, nz_mean, exp_sample='HSC_DESC')
        tr.append(T)

    s_m = sacc.SACC(tr, s_data.binning, s_data.mean)
    return s_m, cov_all

def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

s_mean, cov_noise = get_mean_cov(s_d,N_tracers,Nz_per_tracer)
s0 = NzVec(s_mean)

# total cv covmat
cov_CV = np.zeros((Nz_total,Nz_total))
for i in range(N_tracers):
    # cosmic variance covmat for each tracer
    cov_CV_per_tracer = compute_covmat_cv(cosmo,s_mean.tracers[i].z,s_mean.tracers[i].Nz)
    cov_CV[i*Nz_per_tracer:(i+1)*Nz_per_tracer,i*Nz_per_tracer:(i+1)*Nz_per_tracer] = cov_CV_per_tracer

np.save("cov_CV.npy",cov_CV)

cov_CV = np.load("cov_CV.npy")
# impose smoothness
A_smooth = 0.25#0.5
D = A_smooth**2*obtain_smoothing_D(s_mean,first=True,second=True)

cov_total = cov_noise+cov_CV
P0 = np.linalg.inv(cov_total)
s_new = np.dot(np.dot(np.linalg.inv(P0+D),P0),s0)
error = np.sqrt(np.diag(cov_total))


for i in range(N_tracers):
    plt.figure(figsize=(12,8))
    zs = s_d.tracers[i].z
    for pn in pz_codes:
        n = s_d.tracers[i].extra_cols[pn]
        plt.plot(zs,n,'--',lw=1.5,label=pn)
    plt.errorbar(zs,s_new[i*Nz_per_tracer:(i+1)*Nz_per_tracer],yerr=error[i*Nz_per_tracer:(i+1)*Nz_per_tracer],lw=2.,fmt='',capsize=2.,errorevery=3)
    plt.legend()
    plt.savefig("Nz_"+str(i)+".png")
    plt.close()

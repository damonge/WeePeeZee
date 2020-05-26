import sacc
import numpy as np
import matplotlib.pyplot as plt
import os
from compute_CV_cov import compute_covmat_cv
from smoothness import obtain_smoothing_D
import pyccl as ccl

def get_mean_cov(s_data,Ntr,Nztr,noi_fac):
    # photo-z codes
    pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz']

    # store new tracers and noise covariance
    cov_all = np.zeros((Nztr*Ntr,Nztr*Ntr))
    tr = []
    for i in range(Ntr):
        # get nz for all pz codes
        zs = s_data.tracers[i].z
        nzs = [s_data.tracers[i].Nz / np.sum(s_data.tracers[i].Nz)]
        for pn in pz_codes:
            n = s_data.tracers[i].extra_cols[pn]
            nzs.append(n/np.sum(n))
        nzs = np.array(nzs)

        # get mean and variance
        nz_mean = np.mean(nzs, axis=0)
        nz_var = np.var(nzs,axis=0)
        nz_var_mean = nz_var[nz_mean>0].mean()
        cov = np.diag(noi_fac*nz_var_mean*np.ones(len(zs)))
        cov_all[i*len(zs):(i+1)*len(zs),i*len(zs):(i+1)*len(zs)] = cov
        
        # store new tracers
        T = sacc.Tracer('bin_%d'%i, 'point',
                        zs, nz_mean, exp_sample='HSC_DESC')
        tr.append(T)

    s_m = sacc.SACC(tr, s_data.binning, s_data.mean)
    return s_m, cov_all

def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

def get_smooth_s_and_prior(s_data,cosmo,want_prior,A_smooth=0.25,noi_fac=4.):
    # number of tracers and bins
    Nz_per_tracer = len(s_data.tracers[0].z)
    N_tracers = len(s_data.tracers)
    Nz_total = N_tracers*Nz_per_tracer
    zs = s_data.tracers[0].z

    # obtain the mean of the 4 pz codes with their noise
    s_mean, cov_noise = get_mean_cov(s_data,N_tracers,Nz_per_tracer,noi_fac)
    s0 = NzVec(s_data)

    
    # compute the CV
    if os.path.isfile("cov_CV.npy"):
        print ("!!!!! Loading cached CV covariance matrix !!!!!")
        cov_CV = np.load("cov_CV.npy")
    else:
        # compute cv covmat
        cov_CV = np.zeros((Nz_total,Nz_total))
        for i in range(N_tracers):
            # cosmic variance covmat for each tracer
            cov_CV_per_tracer = compute_covmat_cv(cosmo,s_mean.tracers[i].z,s_mean.tracers[i].Nz)
            cov_CV[i*Nz_per_tracer:(i+1)*Nz_per_tracer,i*Nz_per_tracer:(i+1)*Nz_per_tracer] = cov_CV_per_tracer
        np.save("cov_CV.npy",cov_CV)

    # impose smoothness of first and second derivative
    D = A_smooth**2*obtain_smoothing_D(s_mean,first=True,second=True)

    # compute total covariance of noise
    cov_total = cov_noise+cov_CV

    # compute precision with and without the smoothing matrix D
    P0 = np.linalg.inv(cov_total)
    P = P0+D

    # get the smoothed N(z) for all tracers
    s_smooth = np.dot(np.dot(np.linalg.inv(P0+D),P0),s0)
    print (s0[:10],s_smooth[:10])
    tr = []
    for i in range(N_tracers):
        T = sacc.Tracer('bin_%d'%i, 'point',
                        zs, s_smooth[i*Nz_per_tracer:(i+1)*Nz_per_tracer], exp_sample='HSC_DESC')
        tr.append(T)
    s = sacc.SACC(tr, s_data.binning, s_data.mean)

    # return smooth s (and smooth prior)
    if want_prior:
        return s, P
    else:
        return s


# testing of the code
def main():
    pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz']

    cosmo_params = {'n_s': 0.9649,
                    'sigma8':0.8111,
                    'h':0.6736,
                    'Omega_c':0.264,
                    'Omega_b':0.0493}
    
    # Theory prediction
    cosmo = ccl.Cosmology(**cosmo_params)

    s_d = sacc.SACC.loadFromHDF("data/COADDED/power_spectra_wdpj.sacc")
    s_sm, prior_sm = get_smooth_s_and_prior(s_d,cosmo,want_prior=True,A_smooth=0.25,noi_fac=4.)

    Nz_per_tracer = len(s_d.tracers[0].z)
    N_tracers = len(s_d.tracers)
    Nz_total = N_tracers*Nz_per_tracer
    zs = s_d.tracers[0].z

    error = np.sqrt(np.diag(np.linalg.inv(prior_sm)))
    
    # Plotting
    for i in range(N_tracers):
        plt.figure(figsize=(12,8))
        zs = s_d.tracers[i].z
        for pn in pz_codes:
            n = s_d.tracers[i].extra_cols[pn]
            plt.plot(zs,n,'--',lw=1.5,label=pn)
        plt.errorbar(zs,s_sm.tracers[i].Nz,yerr=error[i*Nz_per_tracer:(i+1)*Nz_per_tracer],lw=2.,fmt='',capsize=2.,errorevery=3)
        plt.legend()
        plt.savefig("Nz_"+str(i)+".png")
        plt.close()

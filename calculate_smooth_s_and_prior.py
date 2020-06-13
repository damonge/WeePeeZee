import sacc
import numpy as np
import matplotlib.pyplot as plt
import os
from compute_CV_cov import compute_covmat_cv
from smoothness import obtain_generalized_D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import pyccl as ccl
from scipy import interpolate

def get_mean_cov(s_data,Ntr,Nztr,noi_fac,upsample):
    # photo-z codes

    pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz']

    # store new tracers and noise covariance
    myNztr=Nztr*upsample
    cov_all = np.zeros((myNztr*Ntr,myNztr*Ntr))
    tr = []
    for i in range(Ntr):
        # get nz for all pz codes
        zs = s_data.tracers[i].z
        nzs = s_data.tracers[i].Nz
        nzs /= np.sum(nzs)
        if (upsample>=1):
            minz,maxz= zs[0],zs[-1]
            newzs = zs[0] + np.arange(myNztr) * (maxz-minz)/(Nztr-1)/upsample
            newnzs = np.zeros(myNztr)
            w=np.where(newzs<=maxz)
            newnzs[w] = [interp1d(zs,nzs, kind='cubic')(newzs[w])]
            nzs = [newnzs]
        else:
            newzs = zs
            nzs = [nzs]

            
            
        for pn in pz_codes:
            n = s_data.tracers[i].extra_cols[pn]
            n /= np.sum(n)
            newn = np.zeros(myNztr)
            newn[w] = interp1d(zs,n, kind='cubic')(newzs[w])
            nzs.append(newn)
        nzs = np.array(nzs)

        # get mean and variance
        nz_mean = np.mean(nzs, axis=0)
        nz_var = np.var(nzs,axis=0)
        # TODO: is this necessary? 
        nz_var = gaussian_filter(nz_var, 2.5*upsample)
        # TODO: is this correct?
        # used to be
        #corr = np.eye(Nztr)
        #sqrtnz_var = np.sqrt(nz_var)
        #cov = noi_fac*np.outer(sqrtnz_var,sqrtnz_var)*corr
        # I think it should be 
        cov = noi_fac*np.diag(nz_var)
        cov_all[i*myNztr:(i+1)*myNztr,i*myNztr:(i+1)*myNztr] = cov
        
        # store new tracers
        T = sacc.Tracer('bin_%d'%i, 'point',
                        newzs, nz_mean, exp_sample='HSC_DESC')
        tr.append(T)
    s_m = sacc.SACC(tr, s_data.binning, s_data.mean)
    return s_m, cov_all

def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

def get_smooth_s_and_prior(s_data,cosmo,noi_fac=4.,A_smooth=1.,dz_thr=0.04,upsample=1,cov_cv=True):
    # number of tracers and bins
    Nz_per_tracer = len(s_data.tracers[0].z)
    N_tracers = len(s_data.tracers)
    Nz_total = N_tracers*Nz_per_tracer
    zs_data = s_data.tracers[0].z

    # obtain the mean of the 4 pz codes with their noise
    s_mean, cov_noise = get_mean_cov(s_data,N_tracers,Nz_per_tracer,noi_fac,upsample)
    zs_mean = s_mean.tracers[0].z
    s0 = NzVec(s_mean)

    if cov_cv:
        # compute the CV
        covfn = "cov_CV_%i.npy"%(upsample)
        if os.path.isfile(covfn):
            print ("!!!!! Loading cached CV covariance matrix !!!!!")
            cov_CV = np.load(covfn)
        else:
            # compute cv covmat
            cov_CV = np.zeros((Nz_total*upsample,Nz_total*upsample))
            for i in range(N_tracers):
                print("Tracer = %i out of %i"%(i,N_tracers-1))
                # cosmic variance covmat for each tracer
                cov_CV_per_tracer = compute_covmat_cv(cosmo,s_data.tracers[i].z,s_data.tracers[i].Nz)
                
                if upsample > 1:
                    cov_CV_up_per_tracer = np.zeros((len(zs_mean),len(zs_mean)))
                    for row in range(Nz_per_tracer):
                        fun = interpolate.interp1d(zs_data,cov_CV_per_tracer[row,:],fill_value="extrapolate")
                        cov_CV_up_per_tracer[row,:] = fun(zs_mean)

                    for col in range(Nz_per_tracer*upsample):
                        fun = interpolate.interp1d(zs_data,cov_CV_up_per_tracer[:len(zs_data),col],fill_value="extrapolate")
                        cov_CV_up_per_tracer[:,col] = fun(zs_mean)
                        
                    cov_CV[i*len(zs_mean):(i+1)*len(zs_mean),i*len(zs_mean):(i+1)*len(zs_mean)] = cov_CV_up_per_tracer
                else:
                    cov_CV[i*len(zs_mean):(i+1)*len(zs_mean),i*len(zs_mean):(i+1)*len(zs_mean)] = cov_CV_per_tracer
            np.save(covfn,cov_CV)
    else:
        cov_CV = 0
    
    # impose smoothness 
    D = obtain_generalized_D(s_mean,A_smooth,dz_thr)

    # compute total covariance of noise
    cov_total = cov_noise +  cov_CV
    # compute precision with and without the smoothing matrix D
    P0 = np.linalg.inv(cov_total)
    P = P0+D

    # get the smoothed N(z) for all tracers
    s_smooth = np.dot(np.dot(np.linalg.inv(P0+D),P0),s0)
    #s_smooth = s0
    
    tr = []
    for i in range(N_tracers):
        T = sacc.Tracer('bin_%d'%i, 'point',
                        zs_mean, s_smooth[i*Nz_per_tracer*upsample:(i+1)*Nz_per_tracer*upsample], exp_sample='HSC_DESC')
        tr.append(T)
    s = sacc.SACC(tr, s_data.binning, s_data.mean)

    # return smooth s and smoothing prior
    return s,P


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

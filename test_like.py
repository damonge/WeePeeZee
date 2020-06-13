import sacc
import copy
import numpy as np
import pyccl as ccl
from HSC_wrapper import get_theory
from calculate_smooth_s_and_prior import get_smooth_s_and_prior

# HSC modules
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from desclss.halo_mod_corr import HaloModCorrection
import yaml

import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

#######################################
###              MY CODE            ###
#######################################
# Names
read = sys.argv[1]# NEWCOV_COADDED
dir_read = "data/"+read

#dir_read = "data/NEWCOV_MARG"
#dir_read = "data/NEWCOV_MARG_nosmooth"
#dir_read = "data/NEWCOV_MARG_largenoise"
#dir_read = "data/NEWCOV_MARG_nosmooth_largenoise"
only_6HOD = bool(int(sys.argv[2]))
test_Tmat = 1
test_zparams = 0
verbose = 0

np.random.seed(300)


# Function to calculate chi2
def print_chi2(di,precision):
    # Report the chi2
    return np.dot(di,np.dot(precision,di))

# Auxiliary function to get the Nzs for all (4) tracers
def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

# Remove this for the actual MCMC run as the sampler makes its own cut TODO
lmax = [2170.58958919, 2515.39193451, 3185.36076391, 4017.39370804]
#lmax = [2000,2000,2600,3200]

# Cosmo params: same as what the MCMC chain fixes for them
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

z_params = {'zshift_bin0':0.000,'zshift_bin1':-0.016,\
            'zshift_bin2':-0.01,'zshift_bin3':0.01,\
            'zwidth_bin0':-0.05,'zwidth_bin1':-0.01,\
            'zwidth_bin2':0.035,'zwidth_bin3':0.05}

z_params0 = {'zshift_bin0':0.000,'zshift_bin1':-0.0,\
            'zshift_bin2':-0.0,'zshift_bin3':0.0,\
            'zwidth_bin0':-0.0,'zwidth_bin1':-0.0,\
            'zwidth_bin2':0.0,'zwidth_bin3':0.0}

# Theory prediction
cosmo = ccl.Cosmology(**cosmo_params)

# Load the data whose precision matrix we would like to modify
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")
s_noi = sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")

# Calculate the smooth s_m = (P0+D)^-1 P0 s0 and smooth prior prior_smo = P0+D
s_m, prior_m = get_smooth_s_and_prior(s_d,cosmo,want_prior=True,A_smooth=0.25,noi_fac=4.)

# TESTING
# Applying l-cuts; remove for MCMC run
s_d.cullLminLmax([0,0,0,0],lmax)

# T matrix
Tmat = np.load("Tmat_NEWCOV_COADDED.npy")

# Halo Model Correction: have checked that this is the same as what is applied for the MCMC run
HMCorrection = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

# Computing Cls directly from the data
cl_theory = get_theory(s_d,s_noi,hod_params,z_params,HMCorrection)

# Get the covariance and precision matrices
N_data = len(s_d.mean.vector)
cov = s_d.precision.getCovarianceMatrix()
prec = np.linalg.inv(cov)

# Difference of N(z) between smooth and jagged
dNz = NzVec(s_m)-NzVec(s_d)

# Slightly improved computation using taylor expansion
cl_theory_taylor = cl_theory + np.dot(Tmat.T,dNz)

# Test how well the Cls are captured by the Tmatrix
if test_Tmat:
    N_tomo = 4
    zs = s_m.tracers[0].z
    nz_size = len(zs)
    cov_m = np.linalg.inv(prior_m)
    s_new = copy.deepcopy(s_m)
    for i in range(N_tomo):
        nz_new = s_new.tracers[i].Nz
        if i == 2:
            gaussian_draw = np.random.multivariate_normal(np.zeros(nz_size), cov_m[i*nz_size:(i+1)*nz_size,i*nz_size:(i+1)*nz_size])
            nz_new[:] += gaussian_draw
    cl_theory_mean = get_theory(s_m,s_noi,hod_params,z_params,HMCorrection)
    cl_theory_new = get_theory(s_new,s_noi,hod_params,z_params,HMCorrection)
    dNz_m = NzVec(s_new)-NzVec(s_m)
    cl_theory_Taylor = cl_theory_mean + np.dot(Tmat.T,dNz_m)
    
    nrows = 5
    ncols = 2
    fs = (ncols*4.,nrows*3.)
    fig, axs = plt.subplots(nrows,ncols,figsize=fs)
    plot_no = 0
    ncomb = nrows*ncols
    for i1, i2, _, ells_binned, ndx in s_d.sortTracers():
        cl_tay = cl_theory_Taylor[ndx]
        cl_ini = cl_theory_mean[ndx]
        cl_per = cl_theory_new[ndx]
        plot_no += 1
        print(plot_no)
        #print(i1,i2,_,ells_binned,ndx)
        ax = axs[(plot_no-1)//ncols,(plot_no-1)%ncols]
        plt.subplot(nrows,ncols,plot_no)
        #plt.plot(ells_binned,cl_ini,'black',lw=3.,label='Initial')
        plt.plot(ells_binned,np.ones(len(cl_ini)),'black',ls='--',lw=3.,label='Initial')
        #plt.plot(ells_binned,cl_tay,'dodgerblue',lw=2.,label='Taylor')
        plt.plot(ells_binned,cl_tay/cl_ini,'dodgerblue',lw=2.,label='Taylor')
        #plt.plot(ells_binned,cl_per,'purple',lw=2.,label='Perturbed')
        plt.plot(ells_binned,cl_per/cl_ini,'purple',lw=2.,label='Perturbed')
        if plot_no == 1: plt.legend()

        if plot_no >= ncomb-ncols+1:
            plt.xlabel(r"$\ell$")
        if plot_no%ncols == 1:
            plt.ylabel(r"$C_\ell$")
        plt.text(0.1, 0.05,'('+str(i1)+','+str(i2)+')',ha='center',va='center',transform=ax.transAxes)
        #plt.yscale('log')
        plt.xscale('log')
    plt.savefig("Cl_comparison.png")
    plt.close()
    
    plt.plot(s_m.tracers[2].z,s_m.tracers[2].Nz,label='initial')
    plt.plot(s_new.tracers[2].z,s_new.tracers[2].Nz,label='perturbed')
    plt.legend()
    plt.xlabel("z")
    plt.ylabel("N(z)")
    plt.savefig("Nz.png")
    plt.close()
    quit()

# Delta Cl
di = s_d.mean.vector - cl_theory_taylor
if verbose: print("Chi2 (naive + Taylor precision) = ",print_chi2(di,prec))
if verbose: print("lnprob (naive + Taylor precision) = ",-0.5*print_chi2(di,prec))

#######################################
###             HSC CODE            ###
#######################################


def cutLranges(saccs, kmax, cosmo, Ntomo, zeff=None, saccs_noise=None):
    zeff = np.zeros(Ntomo)
    for i, t in enumerate(saccs[0].tracers):
        zeff[i] = t.meanZ()
    if verbose: print('zeff = {}.'.format(zeff))

    assert Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
    if verbose: print('Computing lmax according to specified kmax = {}.'.format(kmax))

    lmax = [2170.58958919, 2515.39193451, 3185.36076391, 4017.39370804]

    if Ntomo == 1:
        lmin = [0]
    elif Ntomo == 4:
        lmin=[0,0,0,0]
    else:
        print ("weird Ntomo")

    if verbose: print('lmin = {}, lmax = {}.'.format(lmin, lmax))

    for i, s in enumerate(saccs):
        s.cullLminLmax(lmin, lmax)
        if saccs_noise is not None:
            saccs_noise[i].cullLminLmax(lmin, lmax)

    return saccs, saccs_noise


# HSC part

saccs = [s_d]
saccs_noise = [sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")]
Ntomo = len(saccs[0].tracers) ## number of tomo bins

if only_6HOD:
    config = yaml.load(open("config_6HOD.yaml"))
else:
    config = yaml.load(open("config_6HOD_zparams.yaml"))

fit_params = config['fit_params']
# got a raise exception
for key in fit_params.keys():
    if key in hod_params.keys():
        fit_params[key][1] = hod_params[key]
    else:
        fit_params[key][1] = z_params0[key]

cl_params = config['cl_params']
def_params = config['default_params']
sacc_params = config['sacc_params']
if verbose: print(fit_params)
if verbose: print(cl_params)
if verbose: print(def_params)

saccs, saccs_noise = cutLranges(saccs, sacc_params['kmax'], cosmo=None,
                                Ntomo=Ntomo, saccs_noise=saccs_noise)

noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
for i, s in enumerate(saccs_noise):
    for ii in range(Ntomo):
        binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
        noise[i][ii] = s.mean.vector[binmask]


param_mapping = {}
nparams = len(fit_params.keys())
params = np.zeros((nparams, 4))
for key in fit_params.keys():
    param_mapping[key] = fit_params[key][0]
    params[fit_params[key][0], :] = fit_params[key][1:]
if verbose: print("FIT PARAMS: ",fit_params)

# bias
config['default_params'].update(cl_params['bg'])

th = HSCCoreModule(param_mapping, def_params, cl_params, saccs, noise, HMCorr=HMCorrection)
th.setup()

lik = HSCLikeModule(saccs, temperature=None)
lik.setup()

i = int(sys.argv[3])
pars_min = params[:,1]
pars_max = params[:,2]

# i can vary between -5 and 5
pars = params[:, 0]+i*0.1*params[:, 3]
cl_theory_hsc = th.compute_theory(pars)
if verbose: print("PARS:", pars)


# This part is for testing the Chi2 between the theory with z shifts and widths and no shifts; 
if test_zparams:
    if i != 0: print("The index i (sys.argv[3]) must be set to 0 for this test"); quit()
    if only_6HOD == 0: print("only_6HOD (sys.argv[2]) must be set to 1 for this test since we are comparing with and without the width and shift parameters"); quit()

    # with shift and widths
    for key in fit_params.keys():
        if key in hod_params.keys():
            fit_params[key][1] = hod_params[key]
        else:
            fit_params[key][1] = z_params[key]
    print("FIT PARAMS: ",fit_params)

    param_mapping = {}
    nparams = len(fit_params.keys())
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        param_mapping[key] = fit_params[key][0]
        params[fit_params[key][0], :] = fit_params[key][1:]

    th_zparams = HSCCoreModule(param_mapping, def_params, cl_params, saccs, noise, HMCorr=HMCorrection)
    th_zparams.setup()

    pars = params[:, 0]
    cl_theory_hsc_zparams = th_zparams.compute_theory(pars)
    print("PARS:", pars)

    # Compute the chi2 of the mean posterior HSC paper result (chi2 = 49)
    print(-2.*lik.computeLikelihoodFromCl(cl_theory_hsc_zparams))


    di = cl_theory_hsc_zparams[0]-cl_theory_hsc[0]
    print("chi2 for "+dir_read+" = ",print_chi2(di,prec))
    quit()

# Likelihood part

# This should correspond to the best fit value from the HSC paper, but I don't think it does [-73.5]
lnP = lik.computeLikelihoodFromCl(cl_theory_hsc)
if verbose: print("lnprob theory_hsc = ",lnP)
# Maybe that's because I am not subtracting the noise bias...

# This should be close to 0, I thought as it is just coming from the data [-14974539.5]
lnP = lik.computeLikelihoodFromCl(cl_theory)
if verbose: print("lnprob theory_boryana = ",lnP)

# I believe this should be matching my lnprob [-562.] for cl_theory_taylor, but it's not [-13583893.]
lnP = lik.computeLikelihoodFromCl(cl_theory_taylor)
if verbose: print("lnprob theory_boryana_taylor = ",lnP)

# Minimization part

xtol = 1.e-5
from scipy.optimize import minimize

def chi2(x):
    if np.product(np.logical_and(x>pars_min,x<pars_max)):
        cl = th.compute_theory(x)
        return -2.*lik.computeLikelihoodFromCl(cl)
    else:
        if verbose: print(" <><> infinity with ",x)
        return np.inf

x0 = pars
res = minimize(chi2, x0, method='powell',\
               options={'xtol': xtol, 'disp': True})

print(res.x)

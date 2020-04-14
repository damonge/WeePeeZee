import sacc
import numpy as np
import pyccl as ccl
from modules.halo_mod_corr import HaloModCorrection
from modules.theory_cls import get_theory
from calculate_smooth_s_and_prior import get_smooth_s_and_prior

# HSC modules
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from desclss.halo_mod_corr import HaloModCorrection
import yaml

#######################################
###              MY CODE            ###
#######################################
# Names
dir_read = "data/COADDED"

# Function to calculate chi2
def print_chi2(di,precision):
    # Report the chi2
    return np.dot(di,np.dot(precision,di))

# Auxiliary function to get the Nzs for all (4) tracers
def NzVec(s):
    return np.hstack([t.Nz for t in s.tracers])

# Remove this for the actual MCMC run as the sampler makes its own cut TODO
lmax = [2000,2000,2600,3200]

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


# Theory prediction
cosmo = ccl.Cosmology(**cosmo_params)

# Load the data whose precision matrix we would like to modify
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")

# Calculate the smooth s_m = (P0+D)^-1 P0 s0 and smooth prior prior_smo = P0+D
s_m = get_smooth_s_and_prior(s_d,cosmo,want_prior=False,A_smooth=0.25,noi_fac=4.)

# Applying l-cuts; remove for MCMC run
s_d.cullLminLmax([0,0,0,0],lmax)

# T matrix
Tmat = np.load("Tmat_test_like.npy")

# Halo Model Correction: have checked that this is the same as what is applied for the MCMC run
HMCorrection = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)

# Computing Cls directly from the data
cl_theory = get_theory(hod_params,cosmo_params, s_d, halo_mod_corrector=HMCorrection)

# Get the covariance and precision matrices
N_data = len(s_d.mean.vector)
cov = s_d.precision.getCovarianceMatrix()
prec = np.linalg.inv(cov)

# Difference of N(z) between smooth and jagged
dNz = NzVec(s_m)-NzVec(s_d)

# Slightly improved computation using taylor expansion
cl_theory_taylor = cl_theory + np.dot(Tmat.T,dNz)

# Delta Cl
di = s_d.mean.vector - cl_theory_taylor
print("Chi2 (naive + Taylor precision) = ",print_chi2(di,prec))
print("lnprob (naive + Taylor precision) = ",-0.5*print_chi2(di,prec))

#######################################
###             HSC CODE            ###
#######################################


def cutLranges(saccs, kmax, cosmo, Ntomo, zeff=None, saccs_noise=None):
    zeff = np.zeros(Ntomo)
    for i, t in enumerate(saccs[0].tracers):
        zeff[i] = t.meanZ()
    print('zeff = {}.'.format(zeff))

    assert Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
    print('Computing lmax according to specified kmax = {}.'.format(kmax))

    lmax = [2000,2000,2600,3200]#kmax2lmax(kmax, zeff, cosmo)

    if Ntomo == 1:
        lmin = [0]
    elif Ntomo == 4:
        lmin=[0,0,0,0]
    else:
        print ("weird Ntomo")

    print('lmin = {}, lmax = {}.'.format(lmin, lmax))

    for i, s in enumerate(saccs):
        s.cullLminLmax(lmin, lmax)
        if saccs_noise is not None:
            saccs_noise[i].cullLminLmax(lmin, lmax)

    return saccs, saccs_noise


print("HSC part starts")


saccs = [s_d]
saccs_noise = [sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")]
Ntomo = len(saccs[0].tracers) ## number of tomo bins

config = yaml.load(open("test_like.yaml"))

fit_params = config['fit_params']
# got a raise exception
new_fit_params = fit_params.copy()
for key in fit_params.keys():
    new_fit_params[key][1] = hod_params[key]

cl_params = config['cl_params']
def_params = config['default_params']
sacc_params = config['sacc_params']
print(fit_params)
print(new_fit_params)
print(cl_params)
print(def_params)

saccs, saccs_noise = cutLranges(saccs, sacc_params['kmax'], cosmo=None,
                                Ntomo=Ntomo, saccs_noise=saccs_noise)

noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
for i, s in enumerate(saccs_noise):
    for ii in range(Ntomo):
        binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
        noise[i][ii] = s.mean.vector[binmask]


param_mapping = {}
nparams = len(new_fit_params.keys())
params = np.zeros((nparams, 4))
for key in new_fit_params.keys():
    param_mapping[key] = new_fit_params[key][0]
    params[new_fit_params[key][0], :] = new_fit_params[key][1:]

# bias
config['default_params'].update(cl_params['bg'])

th = HSCCoreModule(param_mapping, def_params, cl_params, saccs, noise, HMCorr=HMCorrection)
th.setup()
lik = HSCLikeModule(saccs, temperature=None)
lik.setup()

i = 1
pars = params[:, 0]+i*0.1*params[:, 3]
cl_theory_hsc = th.compute_theory(pars)

# This should correspond to the best fit value from the HSC paper, but I don't think it does [-73.5]
lnP = lik.computeLikelihoodFromCl(cl_theory_hsc)
print("lnprob theory_hsc = ",lnP)

# Maybe that's because I am not subtracting the noise bias...

# This should be close to 0, I thought as it is just coming from the data [-14974539.5]
lnP = lik.computeLikelihoodFromCl(cl_theory)
print("lnprob theory_boryana = ",lnP)

# I believe this should be matching my lnprob [-562.] for cl_theory_taylor, but it's not [-13583893.]
lnP = lik.computeLikelihoodFromCl(cl_theory_taylor)
print("lnprob theory_boryana_taylor = ",lnP)

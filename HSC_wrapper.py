import sacc
import copy
import numpy as np
import pyccl as ccl
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from desclss.halo_mod_corr import HaloModCorrection
import yaml

def cutLranges(saccs,saccs_noise):
    # ell-ranges
    lmin = [0, 0, 0, 0]
    lmax = [2170.58958919, 2515.39193451, 3185.36076391, 4017.39370804]

    # do the cuts
    for i, s in enumerate(saccs):
        s.cullLminLmax(lmin, lmax)
        if saccs_noise is not None:
            saccs_noise[i].cullLminLmax(lmin, lmax)
    return saccs, saccs_noise


def setup(hod_params,z_params):
    # yaml file
    config = yaml.load(open("test_like.yaml"))

    # set the HOD parameters to the posterior mean from the paper and the shifts and widths to 0
    fit_params = config['fit_params']
    for key in fit_params.keys():
        if key in hod_params.keys():
            fit_params[key][1] = hod_params[key]
        else:
            fit_params[key][1] = z_params[key]

    # read the parameters in the config file
    cl_params = config['cl_params']
    def_params = config['default_params']
    sacc_params = config['sacc_params']

    # set the parameters
    param_mapping = {}
    nparams = len(fit_params.keys())
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        param_mapping[key] = fit_params[key][0]
        params[fit_params[key][0], :] = fit_params[key][1:]

    # update the bias
    config['default_params'].update(cl_params['bg'])

    return params, param_mapping, sacc_params, cl_params, def_params


def _get_cl_theory(s_d,s_n,params,param_mapping,sacc_params,cl_params,def_params,HMCorr,want_Lcuts=False):
    # sacc files for noise and power spectra
    saccs = [s_d]
    saccs_noise = [s_n]
    Ntomo = len(saccs[0].tracers)

    if want_Lcuts:
        # do the ell-cutting
        saccs, saccs_noise = cutLranges(saccs,saccs_noise=saccs_noise)

    # get the noise
    noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
    for i, s in enumerate(saccs_noise):
        for ii in range(Ntomo):
            binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
            noise[i][ii] = s.mean.vector[binmask]


    # compute the theory for these HOD and default parameters
    th = HSCCoreModule(param_mapping, def_params, cl_params, saccs, noise, HMCorr=HMCorr)
    th.setup()
    pars = params[:, 0]#+params[:, 3]
    cl_theory_hsc = th.compute_theory(pars)

    return cl_theory_hsc[0]

def get_lnprob(sacc,cl):
    # compute the likelihood
    # TODO: perhaps setup beforehand
    saccs = [sacc]
    lik = HSCLikeModule(saccs, temperature=None)
    lik.setup()
    cl = [cl]
    return lik.computeLikelihoodFromCl(cl)

def get_chi2(sacc,cl):
    delta = sacc.mean.vector - cl                                                                                                                                                            
    pmatrix = sacc.precision.getPrecisionMatrix()
    chi2 = np.einsum('i,ij,j',delta, pmatrix, delta)
    return chi2
    
def obtain_Tmat(s_d,s_noi,hod_params,z_params,HMCorr,want_Lcuts=False):
    # setup the HSC modules
    params,param_mapping,sacc_params,cl_params,def_params = setup(hod_params,z_params)

    # Number of tracers and z-bins
    Nztr = len(s_d.tracers[0].z)
    Ntomo = len(s_d.tracers)
    Nz = Ntomo*Nztr
    Nd = len(s_d.mean.vector)
    
    # compute numerical derivatives of the Cls with respect to dndz
    delta = 1.e-3
    Tmat = np.zeros((Nz,Nd))
    for i in range(Nz):
        print ("tomo bin, z-sample = %i , %i "%(i//Nztr,i%Nztr))
        s_tmp = copy.deepcopy(s_d)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] += delta
        cl_plus = _get_cl_theory(s_tmp,s_noi,params,param_mapping,sacc_params,cl_params,def_params,HMCorr)
        s_tmp = copy.deepcopy(s_d)
        s_tmp.tracers[i//Nztr].Nz[i % Nztr] -= delta
        cl_minus = _get_cl_theory(s_tmp,s_noi,params,param_mapping,sacc_params,cl_params,def_params,HMCorr)
        Tmat[i,:] = (cl_plus-cl_minus)/(2*delta)
    return Tmat

def get_theory(s_d,s_noi,hod_params,z_params,HMCorr,want_Lcuts=False):
    # setup the HSC modules
    params,param_mapping,sacc_params,cl_params,def_params = setup(hod_params,z_params)

    # compute the power spectrum
    cl_theory = _get_cl_theory(s_d,s_noi,params,param_mapping,sacc_params,cl_params,def_params,HMCorr)
    return cl_theory
    
def main():
    # For testing
    
    # Names
    read = 'NEWCOV_COADDED'
    dir_read = "data/"+read

    # Cosmo params: same as what the MCMC chain fixes for them
    cosmo_params = {'n_s': 0.9649,
                    'sigma8':0.8111,
                    'h':0.6736,
                    'Omega_c':0.264,
                    'Omega_b':0.0493}

    # HOD params: assuming posterior mean from paper
    HOD_params = {'zfid':0.65,
                  'lmmin':11.88, 'lmminp':-0.5,##
                  'sigm_0':0.4, 'sigm_1':0,
                  'm0':5.7, 'm0p':2.5,##
                  'm1':13.08, 'm1p':0.9,##
                  'alpha_0':1, 'alpha_1':0,
                  'fc_0':1., 'fc_1':0.}

    Z_params = {'zshift_bin0':0.000,'zshift_bin1':-0.016,\
                'zshift_bin2':-0.01,'zshift_bin3':0.01,\
                'zwidth_bin0':-0.05,'zwidth_bin1':-0.01,\
                'zwidth_bin2':0.035,'zwidth_bin3':0.05}

    Z_params0 = {'zshift_bin0':0.000,'zshift_bin1':-0.0,\
                'zshift_bin2':-0.0,'zshift_bin3':0.0,\
                'zwidth_bin0':-0.0,'zwidth_bin1':-0.0,\
                'zwidth_bin2':0.0,'zwidth_bin3':0.0}


    # set up the cosmology
    cosmology = ccl.Cosmology(**cosmo_params)

    # Halo Model Correction: have checked that this is the same as what is applied for the MCMC run
    HMCorrection = HaloModCorrection(cosmology, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)
    
    # setup the HSC modules
    params,param_mapping,sacc_params,cl_params,def_params = setup(HOD_params,Z_params0)

    
    # load HSC data sacc files
    s_data = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")
    s_noise = sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")
    
    # compute theory
    cl_theory_hsc = _get_cl_theory(s_data,s_noise,params,param_mapping,sacc_params,cl_params,def_params,HMCorrection)

    # print lnprob, chi2
    print(get_lnprob(s_data,cl_theory_hsc))
    print(get_chi2(s_data,cl_theory_hsc))

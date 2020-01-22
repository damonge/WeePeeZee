import pyccl as ccl
import numpy as np
from .hod import HODProfile
from .hod_funcs_evol_fit import HODParams
from .cl_interpolator import ClInterpolator

def get_theory(hod_params, cosmo_params, sacc_data, halo_mod_corrector=None):
    # Cosmology
    cosmo = ccl.Cosmology(n_s=cosmo_params['n_s'],
                          sigma8=cosmo_params['sigma8'],
                          h=cosmo_params['h'],
                          Omega_c=cosmo_params['Omega_c'],
                          Omega_b=cosmo_params['Omega_b'])

    # Tracers
    tr_g = []
    for i_t, t in enumerate(sacc_data.tracers):
        z = t.z
        nz = t.Nz
        a = 1./(1+z[::-1])
        b = np.ones(len(a))
        g_kernel = ccl.get_density_kernel(cosmo, (z, nz))

        t_g = ccl.Tracer()
        t_g.add_tracer(cosmo=cosmo, kernel=g_kernel, transfer_a=(a, b))
        tr_g.append(t_g)

    #Ell binning
    ls_bin = sacc_data.sortTracers()[0][3]
    ls_w = sacc_data.binning.windows[0].ls
    cli=ClInterpolator(ls_bin)

    # HOD power spectrum
    # HOD functions and profile
    hodpars = HODParams(hod_params, islogm0=True, islogm1=True)
    hodprof = HODProfile(cosmo, hodpars.lmminf, hodpars.sigmf,
                         hodpars.fcf, hodpars.m0f, hodpars.m1f,
                         hodpars.alphaf)
    # Power spectrum
    k_arr = np.logspace(-4.3, 1.5, 256)
    z_arr = np.linspace(0., 3., 50)[::-1]
    a_arr = 1./(1. + z_arr)
    pk_hod_arr = np.array([hodprof.pk(k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in a_arr])
    # Correct for transition regime if needed
    if halo_mod_corrector:
        rk = halo_mod_corrector.rk_interp(k_arr, a_arr)
        pk_hod_arr *= rk
    pk_hod_arr = np.log(pk_hod_arr)
    pk_hod = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_hod_arr, is_logp=True)

    # Compute Cls
    cl_t = np.zeros_like(sacc_data.mean.vector)
    for i1, i2, _, ells_binned, ndx in sacc_data.sortTracers() :
        # At interpolation nodes
        clb = ccl.angular_cl(cosmo, tr_g[i1], tr_g[i2], cli.ls_eval, p_of_k_a=pk_hod)
        # Interpolate on all integer ells
        cls = cli.interpolate_and_extrapolate(ls_w, clb)
        # Convolve with windows
        cls_conv = np.zeros(ndx.shape[0])
        for j in range(ndx.shape[0]):
            cls_conv[j] = sacc_data.binning.windows[ndx[j]].convolve(cls)
        cl_t[ndx] = cls_conv

    return cl_t

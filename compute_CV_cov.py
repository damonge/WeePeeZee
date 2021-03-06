import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import jv
from scipy.integrate import simps

def get_nz_from_photoz_bins(weights_fn,zp_code,zp_ini,zp_end,zt_edges,zt_nbins):
    cat = fits.open(weights_fn)[1].data
    # Select galaxies in photo-z bin
    sel = (cat[zp_code] <= zp_end) & (cat[zp_code]>zp_ini)
    # Effective number of galaxies
    ngal = len(cat) * np.sum(cat['weight'][sel])/np.sum(cat['weight'])
    # Make a normalized histogram
    pz,z_bins = np.histogram(cat['PHOTOZ'][sel], # 30-band photo-zs
                             bins = zt_nbins, # Number of bins
                             range = zt_edges, # Range in z_true
                             weights = cat['weight'][sel],# Color-space weights
                             density = True)
    nz = ngal*pz

    zs = cat['PHOTOZ'][sel]
    we = cat['weight'][sel]
    mean = np.sum(zs*we)/np.sum(we)
    sigma = np.sqrt(np.sum(zs**2*we)/np.sum(we)-mean**2)
    return nz, z_bins, ngal, mean, sigma


def compute_covmat_cv(cosmo,zm,dndz):
    # Area COSMOS
    area_deg2 = 1.7
    area_rad2 = area_deg2*(np.pi/180)**2
    theta_rad = np.sqrt(area_rad2/np.pi)
    # TODO: Where do we get the area

    # Bin widths
    dz = np.mean(zm[1:]-zm[:-1])
    # Number of galaxies in each bin
    dn = dndz*dz
    # z bin edges
    zb = np.append((zm-dz/2.),zm[-1]+dz/2.)
    
    # Comoving distance to bin edges
    chis = ccl.comoving_radial_distance(cosmo,1./(1+zb))
    # Mean comoving distance in each bin
    chi_m = 0.5*(chis[1:]+chis[:-1])
    # Comoving distance width
    dchi = chis[1:]-chis[:-1]
    # Disc radii
    R_m = theta_rad*chi_m
    # Galaxy bias
    b_m = 0.95/ccl.growth_factor(cosmo,1./(1+zm))
    # Growth rate (for RSDs)
    f_m = ccl.growth_rate(cosmo,1./(1+zm))
    # Transverse k bins
    n_kt = 512
    # Parallel k bins
    n_kp = 512

    plt.plot(zm,dn,'b-')
    plt.savefig('N_z.png')
    plt.close()
    
    # Transverse modes
    kt_arr = np.geomspace(0.00005,10.,n_kt)
    # Parallel modes
    kp_arr = np.geomspace(0.00005,10.,n_kp)
    # Total wavenumber
    k_arr = np.sqrt(kt_arr[None,:]**2+kp_arr[:,None]**2)
    # B.H. changed a to float(a)
    pk_arr = np.array([(b+f*kp_arr[:,None]**2/k_arr**2)**2*
                     ccl.nonlin_matter_power(cosmo,k_arr.flatten(),float(a)).reshape([n_kp,n_kt])\
                     for a,b,f in zip(1./(1+zm),b_m,f_m)])


    window = np.array([2*jv(1,kt_arr[None,:]*R)/(kt_arr[None,:]*R)*np.sin(0.5*kp_arr[:,None]*dc)/\
                     (0.5*kp_arr[:,None]*dc) for R,dc in zip(R_m,dchi)])

    # Estimating covariance matrix
    # avoiding getting 0s in covariance
    eps  =  0.0
    # Changed from dn to dndz B.H. and A.S. TODO: Check
    print("covmat_cv...")
    covmat_cv  =  np.array([[(ni+eps)*(nj+eps)*covar(i,j,window,pk_arr,chi_m,kp_arr,kt_arr) \
                             for i,ni in enumerate(dndz)] \
                            for j,nj in enumerate(dndz)])
    return covmat_cv

def covar(i,j,window,pk_arr,chi_m,kp_arr,kt_arr):
    # Windows
    win_i = window[i]
    win_j = window[j]
    # Power spectrum
    pk_ij = np.sqrt(pk_arr[i]*pk_arr[j])
    # Cosine term
    dchi_ij = chi_m[i]-chi_m[j]
    cos_ij = np.cos(kp_arr*dchi_ij)
    # First integrand
    integrand_pt = (kt_arr[None,:]**2*win_i*win_j*pk_ij)/(2*np.pi**2)
    # Integrate over k_transverse
    integrand_p = simps(integrand_pt,axis=1,x=np.log(kt_arr))*cos_ij*kp_arr
    # Integrate over k_parallel
    integral = simps(integrand_p,x = np.log(kp_arr))
    return integral

def plot_corrmat(covmat,N_zsamples_theo):
    plt.figure()
    plt.imshow(covmat/np.sqrt(np.diag(covmat)[None,:]*np.diag(covmat)[:,None])-np.diag(np.ones(N_zsamples_theo)),interpolation='nearest')
    plt.colorbar()
    plt.savefig("covmat.png")
    plt.close()

def main():
    # Radial bins
    N_zsamples_theo = 50
    z_ini_sample = 0.
    z_end_sample = 2.
    z_bin_ini = .5
    z_bin_end = .75

    # This version is the old one TODO: update
    covmat_cv = compute_covmat_cv(z_bin_ini,z_bin_end,z_ini_sample,z_end_sample,N_zsamples_theo)
    plot_corrmat(covmat_cv,N_zsamples_theo)

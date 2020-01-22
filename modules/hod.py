import numpy as np
import pyccl as ccl
from scipy.special import erf,sici

def concentration_duffy(halo_mass,a,is_Dmatter=False,is_D500=False) :
    """
    Concentration-Mass relation from 0804.2486
    Extended to Delta=500 (critical density)

    .. note:: Returns A*(halo_mass/M_pivot)**B/a**C
              where (A,B,C) depend on the mass definition
              and M_pivot=1E12 M_sun/h

    Arguments
    ---------
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    is_D500 : boolean
        If `True`, return the extension of the original Duffy et al. relation for Delta=500.

    Returns
    -------
    float or array_like : The halo concentration.
    """
    m_pivot=2.78164E12 #Pivot mass in M_sun

    if is_D500 :
        A_Delta=3.67; B_Delta=-0.0903; C_Delta=-0.51;
    else : #2nd row in Table 1 of Duffy et al. 2008
        if is_Dmatter :
            A_Delta=10.14; B_Delta=-0.081; C_Delta=-1.01;
        else :
            A_Delta=5.71; B_Delta=-0.084; C_Delta=-0.47;
    
    return A_Delta*(halo_mass/m_pivot)**B_Delta/a**C_Delta

def r_Delta(cosmo, halo_mass, a, Delta=200, is_matter=False) :
    """
    Calculate the reference radius of a halo.

    .. note:: this is R=(3M/(4*pi*rho_c(a)*Delta))^(1/3), where rho_c is the critical
              matter density

    Arguments
    ---------
    cosmo : ``pyccl.Cosmology`` object
        Cosmological parameters.
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    Delta : float
        Overdensity parameter.

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    omega_factor=1.
    if is_matter :
        omega_factor=ccl.omega_x(cosmo,a,'matter')
    prefac=Delta*omega_factor*1.16217766E12*(cosmo['h']*ccl.h_over_h0(cosmo,a))**2
    return (halo_mass/prefac)**(1./3.)


class HODProfile(object) :
    """
    Implements an HOD profile as described in https://arxiv.org/abs/1601.05779 (Eqs. 22 and 23).
    """
    def __init__(self,cosmo,lmmin_f,sigm_f,fc_f,m0_f,m1_f,
                 alpha_f,delta_mass=200.,is_delta_matter=True,
                 verbose=False) :
        """
        cosmo : CCL cosmology object
        lmmin_f : function returning log10(M_min)(z)
        sigm_f : function returning sigma_lnM(z)
        fc_f : function returning f_central(z)
        m0_f : function returning M_0(z)
        m1_f : function returning M_1(z)
        alpha_f : function returning alpha(z)
        """

        self.cosmo = cosmo
        self.lmmin_f=lmmin_f
        self.sigm_f=sigm_f
        self.fc_f=fc_f
        self.m0_f=m0_f
        self.m1_f=m1_f
        self.alpha_f=alpha_f
        self.delta_mass=delta_mass
        self.is_delta_matter=is_delta_matter
        if verbose:
            print('HODProfile called with {}.'.format(cosmo))

    def n_cent(self,z,m) :
        """
        Number of central galaxies
        """
        lmmin=self.lmmin_f(z)
        sigm=self.sigm_f(z)
        return 0.5*(1+erf((np.log10(m)-lmmin)/sigm))

    def n_sat(self,z,m) :
        """
        Number of satellite galaxies
        """
        m0=self.m0_f(z)
        m1=self.m1_f(z)
        alpha=self.alpha_f(z)
        f1=lambda x: np.zeros_like(x)
        f2=lambda x: ((x-m0)/m1)**alpha
        return np.piecewise(m,[m<=m0,m>m0],[f1,f2])

    def n_tot(self,z,m) :
        """
        Total number of galaxies
        """
        fc=self.fc_f(z)
        return self.n_cent(z,m)*(fc+self.n_sat(z,m))

    def u_sat(self,z,m,k) :
        """
        Satellite density profile
        """
        a=1./(1+z)
        r200=r_Delta(self.cosmo,m,a,self.delta_mass,is_matter=self.is_delta_matter)*(1+z)
        c200=concentration_duffy(m,a,is_Dmatter=self.is_delta_matter,is_D500=(self.delta_mass==500))
        norm=1./(np.log(1+c200)-c200/(1+c200))
        xarr=k[:,None]*r200[None,:]/c200[None,:]
        opcx=xarr*(1+c200[None,:])
        si,ci=sici(xarr)
        siopc,ciopc=sici(opcx)
        sinx=np.sin(xarr)
        cosx=np.cos(xarr)
        return norm[None,:]*(sinx*(siopc-si)+cosx*(ciopc-ci)-np.sin(c200[None,:]*xarr)/opcx)

    def pk(self,k,a,lmmin=6.,lmmax=17.,nlm=256,return_decomposed=False) :
        """
        Returns power spectrum at redshift `z` sampled at all values of k in `k`.

        k : array of wavenumbers in CCL units
        a : scale factor
        lmmin, lmmax, nlm : mass edges and sampling rate for mass integral.
        return_decomposed : if True, returns 1-halo, 2-halo, bias, shot noise and total (see below for order).
        """
        z = 1./a - 1.
        marr=np.logspace(lmmin,lmmax,nlm); dlm=np.log10(marr[1]/marr[0])
        u_s=self.u_sat(z,marr,k)
        hmf=ccl.massfunc(self.cosmo,marr,a);
        hbf=ccl.halo_bias(self.cosmo,marr,a);
        rhoM=ccl.rho_x(self.cosmo,a,"matter",is_comoving=True)
        n0_1h=(rhoM-np.sum(hmf*marr)*dlm)/marr[0]
        n0_2h=(rhoM-np.sum(hmf*hbf*marr)*dlm)/marr[0]

        #Number of galaxies
        fc=self.fc_f(z)
        ngm=self.n_tot(z,marr)
        ncm=self.n_cent(z,marr)
        nsm=self.n_sat(z,marr)

        #Number density
        ng=np.sum(hmf*ngm)*dlm+n0_1h*ngm[0]
        if ng<=1E-16 : #Make sure we won't divide by 0
            return None

        #Bias
        b_hod=np.sum((hmf*hbf*ncm)[None,:]*(fc+nsm[None,:]*u_s[:,:]),axis=1)*dlm+n0_2h*ncm[0]*(fc+nsm[0]*u_s[:,0])
        b_hod/=ng

        #1-halo
        #p1h=np.sum((hmf*ncm**2)[None,:]*(fc+nsm[None,:]*u_s[:,:])**2,axis=1)*dlm+n0_1h*(ncm[0]*(fc+nsm[0]*u_s[:,0]))**2
        p1h=np.sum((hmf*ncm)[None,:]*(2*fc*nsm[None,:]*u_s[:,:]+(nsm[None,:]*u_s[:,:])**2),axis=1)*dlm+n0_1h*ncm[0]*(2*fc*nsm[0]*u_s[:,0]+(nsm[0]*u_s[:,0])**2)
        p1h/=ng**2

        #2-halo
        p2h=b_hod**2*ccl.linear_matter_power(self.cosmo,k,a)

        if return_decomposed :
            return p1h+p2h,p1h,p2h,np.ones_like(k)/ng,b_hod
        else :
            return p1h+p2h

    def pk_gm(self,k,a,lmmin=6.,lmmax=17.,nlm=256,return_decomposed=False) :
        """
        Returns galaxy-matter power spectrum at a single scale-factor a for array of wave vectors k.
        :param k: wave vector array
        :param a: single scale factor value
        :param lmmin: Mmin for HOD integrals
        :param lmmax: Mmax for HOD integrals
        :param nlm: sampling rate for mass integral
        :param return_decomposed: boolean tag, if True return 1h, 2h power spectrum separately
        :return:
        """

        z = 1./a - 1.
        marr=np.logspace(lmmin,lmmax,nlm); dlm=np.log10(marr[1]/marr[0])
        u_nfw=self.u_sat(z,marr,k)
        hmf=ccl.massfunc(self.cosmo,marr,a);
        hbf=ccl.halo_bias(self.cosmo,marr,a);
        rhoM=ccl.rho_x(self.cosmo,a,"matter",is_comoving=True)
        n0_1h=(rhoM-np.sum(hmf*marr)*dlm)/marr[0]
        n0_2h=(rhoM-np.sum(hmf*hbf*marr)*dlm)/marr[0]
        # n0_2h = (rhoM - np.sum(hmf*hbf*marr)*dlm)

        #Number of galaxies
        fc=self.fc_f(z)
        ngm=self.n_tot(z,marr)
        ncm=self.n_cent(z,marr)
        nsm=self.n_sat(z,marr)

        #Number density
        ng=np.sum(hmf*ngm)*dlm+n0_1h*ngm[0]
        if ng<=1E-16 : #Make sure we won't divide by 0
            return None

        #Bias
        b_hod=np.sum((hmf*hbf*ncm)[None,:]*(fc+nsm[None,:]*u_nfw[:,:]),axis=1)*dlm \
              + n0_2h*ncm[0]*(fc+nsm[0]*u_nfw[:,0])
        b_m = np.sum((hmf*hbf)[None,:]*marr[None,:]*u_nfw[:,:],axis=1)*dlm + n0_2h*u_nfw[:,0]

        b_hod/=ng
        b_m /= rhoM

        #1-halo
        #p1h=np.sum((hmf*ncm**2)[None,:]*(fc+nsm[None,:]*u_s[:,:])**2,axis=1)*dlm+n0_1h*(ncm[0]*(fc+nsm[0]*u_s[:,0]))**2
        p1h=np.sum((hmf*ncm)[None,:]*(fc+nsm[None,:]*u_nfw[:,:])*marr*u_nfw[:,:],axis=1)*dlm \
            + n0_1h*ncm[0]*(fc + nsm[0]*u_nfw[:,0])*u_nfw[:,0]
        p1h/=ng*rhoM

        #2-halo
        p2h=b_hod*b_m*ccl.linear_matter_power(self.cosmo,k,a)

        if return_decomposed :
            return p1h+p2h,p1h,p2h,np.ones_like(k)/ng,b_hod
        else :
            return p1h+p2h

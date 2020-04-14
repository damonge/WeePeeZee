import numpy as np
from scipy.interpolate import interp1d

class ClInterpolator(object):
    def __init__(self,lb,nrb=3,nb_dex_extrap_lo=10,kind='cubic'):
        """Interpolator for angular power spectra
        lb : central bandpower ells
        nrb : re-binning factor for ells within the range of the bandpowers
        nb_dex_extrap_lo : number of ells per decade for ells below the range of the bandpowers
        kind : interpolation type

        Extrapolation at high ell will be done assuming a power-law behaviour,
        with a power-law index estimated from the last two elements of the power spectrum.

        Once initialized, ClInterpolator.ls_eval holds the multipole values at which the
        power spectra should be estimated.
        """

        # Ells below the rannge
        # B.H. included int casting
        ls_pre=np.geomspace(2, lb[0],int(nb_dex_extrap_lo*np.log10(lb[0]/2.)))
        # Ells in range
        ls_mid=(lb[:-1,None]+(np.arange(nrb)[None,:]*np.diff(lb)[:,None]/nrb)).flatten()[1:]
        # Ells above range
        ls_post = np.geomspace(lb[-1], 2*lb[-1], 50)

        self.ls_eval = np.concatenate((ls_pre, ls_mid, ls_post))

        # Interpolation type
        self.kind = kind

    def interpolate_and_extrapolate(self,ls,clb):
        """Go from a C_ell estimated in a few ells to one estimated in a
        finer grid of ells.

        ls : finer grid of ells
        clb : power spectra evaluated at self.ls_eval

        returns : power spectrum evaluated at ls
        """

        # Ells in range
        ind_good = np.where(ls<=self.ls_eval[-1])[0]
        ind_bad = np.where(ls>self.ls_eval[-1])[0]
        clret = np.zeros(len(ls))
        cli = interp1d(self.ls_eval,clb,kind=self.kind,fill_value=0,bounds_error=False)
        clret[ind_good] = cli(ls[ind_good])

        # Extrapolate at high ell
        clret[ind_bad] = clb[-1]*(ls[ind_bad]/self.ls_eval[-1])**-1.05

        return clret

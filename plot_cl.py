import numpy as np
import matplotlib.pyplot as plt
import sacc

dir_read = "data/NEWCOV_COADDED"
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")
s_noi = sacc.SACC.loadFromHDF(dir_read+"/noi_bias.sacc")
shot_noise = s_noi.mean.vector
cov = s_d.precision.getCovarianceMatrix()
upsample = 3
lw = 2.5

import plotparams
plotparams.buba()

import distinct_colours
cs = distinct_colours.get_distinct(4)

zaru = np.load("data_figs/z_arr_up.npy")
zar = np.load("data_figs/z_arr.npy")
Nz = np.load("data_figs/Nz.npy")
Nz_s = np.load("data_figs/Nz_s.npy")
NzP = np.load("data_figs/Nz_p.npy")


fig, ax = plt.subplots(4,1, facecolor="w",
          gridspec_kw={"hspace": 0.0},
          figsize=(14, 10))
for i in range(4):
    li = i*100
    hi = li+100
    ax[i].plot(zar,Nz[li:hi],lw=lw,c=cs[0])
    ax[i].plot(zaru,Nz_s[li*upsample:hi*upsample],ls='--',lw=lw,c=cs[1])
    ax[i].plot(zaru,NzP[li*upsample:hi*upsample],ls='-',lw=lw,c=cs[2])
    ax[i].set_xlim(0,2.6)
    ax[i].set_ylabel(r'$N_%i (z)$'%(i+1))

    if i < 3:
        ax[i].xaxis.set_ticklabels([])
ax[3].set_xlabel(r'$z$')
ax[0].plot([],[],cs[0],label='original')
ax[0].plot([],[],ls='--',c=cs[1],label='smoothed')
ax[0].plot([],[],cs[2],label='perturbed')
ax[0].legend(ncol=3)
plt.savefig('Paper/Nzs.pdf')
plt.close()


cl_theory = np.load("data_figs/cl_theory.npy")-shot_noise
cl_theory_s = np.load("data_figs/cl_theory_s.npy")-shot_noise
cl_theory_perturbed = np.load("data_figs/cl_theory_perturbed.npy")-shot_noise
cl_theory_taylor = np.load("data_figs/cl_theory_taylor.npy")-shot_noise

fig, ax = plt.subplots(4,4, facecolor="w",
        gridspec_kw={"hspace": 0., "wspace":0.4},#0.
        figsize=(23, 16))
data = s_d.mean.vector-shot_noise
err = np.sqrt(cov.diagonal())

for i in range(4):
    for j in range (4):
        cax = ax[j,i]
        if j >= i:
            ndx = s_d.ilrange(i,j)
            ell = s_d.binning.binar['ls'][ndx]
            well = ell**1.2
            cax.plot(ell,well*cl_theory[ndx],c='white',label='$%i^{\mathrm{bin}},%i^{\mathrm{bin}}$'%(i+1,j+1))
            cax.plot(ell,well*cl_theory[ndx],c=cs[0])
            cax.plot(ell,well*cl_theory_s[ndx],c=cs[1])
            cax.plot(ell,well*cl_theory_perturbed[ndx],c=cs[2])
            cax.plot(ell,well*cl_theory_taylor[ndx],c=cs[3],ls=':')
            # HSC data
            #cax.errorbar(ell,data[ndx]*well,yerr=err[ndx]*well,fmt='ko-')
            '''
            # If no space between subplots, i.e. wspace = 0.
            if i > 0:
                cax.yaxis.set_ticklabels([])
                cax.set_ylim(ax[j,0].get_ylim()) # lol, makes things empty -- too different
            else:
                cax.set_ylabel(r'$C_{\ell} \ \ell^{1.2}$')
            '''
            # if including wspace
            if i == 0: cax.set_ylabel(r'$C_{\ell} \ \ell^{1.2}$')
            if j == 3:
                cax.set_xlabel(r'$\ell$')
            else:
                cax.xaxis.set_ticklabels([])
            cax.legend(frameon=False,loc='upper right')
        else:
            cax.yaxis.set_ticklabels([])
            cax.xaxis.set_ticklabels([])

nbins=4
for b1 in range(nbins) :
    for b2 in range(nbins) :
        if b2<b1 :
            ax[b2,b1].axis('off')
            
ax[0,3].plot([],[],cs[0],label=r'original $N(z)$, $C_\ell$')
ax[0,3].plot([],[],cs[1],label=r'smoothed $N(z)$, $C_\ell$')
ax[0,3].plot([],[],cs[2],label=r"perturbed $N(z)$, $C_\ell'$")
ax[0,3].plot([],[],cs[3],ls=':',label=r'perturbed $N(z)$, $C_\ell + {\sf T} \Delta \mathbf{N}$')
# HSC data
#ax[0,3].errorbar([],[],yerr=[],fmt='ko-',label=r'HSC data')
ax[0,3].legend(frameon=False)
plt.savefig('Paper/Cls.pdf')
plt.close()

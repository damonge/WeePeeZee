import sacc
import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

def cov2corr(covmat,nodiag=True):
    if nodiag:
        corrmat = covmat/(np.sqrt(np.diag(covmat)[None,:]*np.diag(covmat)[:,None]))-np.diag(np.ones(covmat.shape[0]))
        corrmat[np.diag(covmat)[None,:]*np.diag(covmat)[:,None] == 0.] = 0.
    else:
        corrmat = covmat/np.sqrt(np.diag(covmat)[None,:]*np.diag(covmat)[:,None])
    return corrmat
    
def plot2d(data,which,logscale=False,lim=None,aspect=1,save=True,levels=None,clip=0,clbar=True,cm=None,label=None,labsize=14,extent=None,ticksize=12,*args):
    # origin at upper left
    if label is not None:
        xlabel, ylabel = label
    else:
        xlabel, ylabel = '',''
    Nx = data.shape[0]
    Ny = data.shape[1]
    arr = data[clip:Nx-clip,clip:Ny-clip]
    if type(lim) is list:
        limmin,limmax = lim
    elif lim==None:
        limmin=None
        limmax = None
    else:
        limmin=-lim
        limmax = lim
    if logscale: arr[arr==0.] = 1.e-14; arr = np.log10(np.abs(arr))
    im = plt.imshow(arr,interpolation="none",vmin=limmin,vmax=limmax,cmap=cm,extent=extent,*args)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_aspect(aspect)
    plt.colorbar(im,fraction=0.1, pad=0.02)
    if save: plt.savefig("Paper/"+which+".pdf"); plt.close()

def plot_evals(data,which):
    Nx = data.shape[0]
    evals,evecs = (np.linalg.eig(data))
    evals_sorted = np.sort(evals)
    assert evals_sorted[0] > 0., "The matrix is not positive definite"
    plt.plot(np.arange(Nx),evals_sorted,label=which)
    plt.yscale('log')

# Names
dir_read = "NEWCOV_COADDED"
dir_write = "NEWCOV_MARG"
upsample = 1

# Load the original and marginalized data 
s_d = sacc.SACC.loadFromHDF("data/"+dir_read+"/power_spectra_wdpj.sacc")
s_m = sacc.SACC.loadFromHDF("data/"+dir_write+"/power_spectra_wdpj.sacc")

# Get their cov matrices
cov_d = s_d.precision.getCovarianceMatrix()
cov_m = s_m.precision.getCovarianceMatrix()

# get the cls
Cl_fid = s_d.mean.vector
Cl_marg = s_m.mean.vector

# redshift array
zs = s_d.tracers[0].z
i_bin = 0

# get the ells
i, j = i_bin,0
ndx = s_d.ilrange(i,j)
ell = s_d.binning.binar['ls'][ndx]
ndx = ndx[0]

# labels
xlabel = r'$z$'
ylabel = r'$\ell$'
cm = 'Reds'

want_evals = 0
if want_evals:
    # Plot the eig vals
    plot_evals(cov_d,'Full 14-parameter')
    plot_evals(cov_m,'Marginalized')
    plt.legend()
    plt.savefig("eigs.png")
    plt.close()

want_cov = 0
if want_cov:
    # Get the corr matrices
    corr_d = cov2corr(cov_d)
    corr_m = cov2corr(cov_m)

    label = [xlabel,xlabel]
    
    # Plot the corr mats
    plot2d(corr_d,'corr_data',extent=[zs[0],zs[-1],zs[-1],zs[0]],label=label,cm=cm)
    #plot2d(corr_m,'corr_marg')
    plot2d(corr_m-corr_d,'corr_diff',extent=[zs[0],zs[-1],zs[-1],zs[0]],label=label,cm=cm)

want_CV = 0
if want_CV:
    # Plot the cosmic variance correlation matrix
    #cov_CV = np.load("cov_CV_%i.npy"%(upsample))
    cov_CV = np.load("og_npy/cov_CV_COADDED.npy")
    corr_CV = cov2corr(cov_CV,nodiag=True)
    label = [xlabel,xlabel]
    #plot2d(corr_CV,'corr_CV')
    plot2d(corr_CV[:(i_bin+1)*100*upsample,:(i_bin+1)*100*upsample],'corr_CV_%i'%(i_bin),extent=[zs[0],zs[-1],zs[-1],zs[0]],label=label,cm=cm)

want_Tmat = 1
if want_Tmat:
    # Plot the T matrix
    Tmat = np.load("Tmat_"+dir_read+"_%i.npy"%(upsample)).T
    factor = 20

    #plt.figure(figsize=(400.*upsample/factor,94./factor))
    #plot2d(Tmat,'Tmat_log',logscale=True)

    #plt.figure(figsize=(400.*upsample/factor,94./factor))
    #plot2d(Tmat/Cl_fid[:,None],'Tmat_to_Cl_fid')

    plt.subplots(2,2,figsize=(14,8))#,gridspec_kw={"hspace": 0.9, "wspace":0.9})

    asp = 1./800.
    
    j = 0; plt.subplot(2,2,j+1); label = [r'$z$',r'$\ell[%i,%i]$'%(i_bin+1,j+1)]
    ndx = s_d.ilrange(i_bin,j)[0]; ell = s_d.binning.binar['ls'][ndx]
    plot2d(Tmat[ndx,(i_bin)*100*upsample:(i_bin+1)*100*upsample]/Cl_fid[ndx,None],'Tmat_%i_to_Cl_fid'%(i_bin),extent=[zs[0],zs[-1],ell[-1],ell[0]],label=label,cm=cm,aspect=asp,save=False)
    j = 1; plt.subplot(2,2,j+1); label = [r'$z$',r'$\ell[%i,%i]$'%(i_bin+1,j+1)]
    ndx = s_d.ilrange(i_bin,j)[0]; ell = s_d.binning.binar['ls'][ndx]
    plot2d(Tmat[ndx,(i_bin)*100*upsample:(i_bin+1)*100*upsample]/Cl_fid[ndx,None],'Tmat_%i_to_Cl_fid'%(i_bin),extent=[zs[0],zs[-1],ell[-1],ell[0]],label=label,cm=cm,aspect=asp,save=False)
    j = 2; plt.subplot(2,2,j+1); label = [r'$z$',r'$\ell[%i,%i]$'%(i_bin+1,j+1)]
    ndx = s_d.ilrange(i_bin,j)[0]; ell = s_d.binning.binar['ls'][ndx]
    plot2d(Tmat[ndx,(i_bin)*100*upsample:(i_bin+1)*100*upsample]/Cl_fid[ndx,None],'Tmat_%i_to_Cl_fid'%(i_bin),extent=[zs[0],zs[-1],ell[-1],ell[0]],label=label,cm=cm,aspect=asp,save=False)
    j = 3; plt.subplot(2,2,j+1); label = [r'$z$',r'$\ell[%i,%i]$'%(i_bin+1,j+1)]
    ndx = s_d.ilrange(i_bin,j)[0]; ell = s_d.binning.binar['ls'][ndx]
    plot2d(Tmat[ndx,(i_bin)*100*upsample:(i_bin+1)*100*upsample]/Cl_fid[ndx,None],'Tmat_%i_to_Cl_fid'%(i_bin),extent=[zs[0],zs[-1],ell[-1],ell[0]],label=label,cm=cm,aspect=asp,save=False)
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.subplots_adjust(hspace=0.2,wspace=0.5)
    plt.savefig("Paper/Tmat.pdf")

    quit()
    # Show Tmat for just the first tomo bin
    factor = 5
    plt.figure(figsize=(16,5))
    #plt.figure(figsize=(100.*upsample/factor,11./factor))
    plot2d(Tmat[ndx,(i_bin)*100*upsample:(i_bin+1)*100*upsample]/Cl_fid[ndx,None],'Tmat_%i_to_Cl_fid'%(i_bin),extent=[zs[0],zs[-1],ell[-1],ell[0]],label=label,cm=cm,aspect=1./2000)


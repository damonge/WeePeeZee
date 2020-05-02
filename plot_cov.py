import sacc
import numpy as np
import matplotlib.pyplot as plt

def cov2corr(cov):                                                                                              
    d = np.diag(cov)
    stddev = np.sqrt(d)
    corr = cov.copy()*0.
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):                                                                           
            corr[i,j] = cov[i,j]/stddev[i]/stddev[j]
    return corr

def cov2corr(covmat):
    corrmat = covmat/np.sqrt(np.diag(covmat)[None,:]*np.diag(covmat)[:,None])-np.diag(np.ones(covmat.shape[0]))
    return corrmat
    
def plot2d(data,which,lim=None,levels=None,clip=0,clbar=True,cm=None,label=None,labsize=14,extent=None,ticksize=12,*args):
    # origin at upper left
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
    plt.imshow(arr,interpolation="none",vmin=limmin,vmax=limmax,cmap=cm,extent=extent,*args)
    plt.colorbar()                                                            
    plt.savefig(which+".png")
    plt.close()

def plot_evals(data,which):
    Nx = data.shape[0]
    evals,evecs = (np.linalg.eig(data))
    evals_sorted = np.sort(evals)
    assert evals_sorted[0] > 0., "The matrix is not positive definite"
    plt.plot(np.arange(Nx),evals_sorted,label=which)
    plt.yscale('log')
    
# Names
#dir_read = "data/COADDED"
dir_read = "NEWCOV_COADDED"
#dir_write = "data/MARG"
dir_write = "NEWCOV_MARG"

# Load the original and marginalized data 
s_d = sacc.SACC.loadFromHDF("data/"+dir_read+"/power_spectra_wdpj.sacc")
s_m = sacc.SACC.loadFromHDF("data/"+dir_write+"/power_spectra_wdpj.sacc")

# Get their cov matrices
cov_d = s_d.precision.getCovarianceMatrix()
cov_m = s_m.precision.getCovarianceMatrix()

# Plot the eig vals
plot_evals(cov_d,'eig_data')
plot_evals(cov_m,'eig_marg')
plt.legend()
plt.savefig("eigs.png")
plt.close()

# Get the corr matrices
corr_d = cov2corr(cov_d)
corr_m = cov2corr(cov_m)

# Plot the corr mats
plot2d(corr_d,'corr_data')
plot2d(corr_m,'corr_marg')

# Plot the cosmic variance correlation matrix
cov_CV = np.load("cov_CV.npy")
corr_CV = cov2corr(cov_CV)
plot2d(corr_CV,'corr_CV')

# Plot the cosmic variance correlation matrix
Tmat = np.load("Tmat_"+dir_read+".npy").T
factor = 40
plt.figure(figsize=(400./factor,94./factor))
plot2d(Tmat,'Tmat')

# Show Tmat for just the first tomo bin
factor = 20
plt.figure(figsize=(100./factor,94./factor))
plot2d(Tmat[:,:100],'Tmat_0')

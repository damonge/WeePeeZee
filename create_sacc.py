import sacc
import sys
import numpy as np

# Note to self and reader -> this should technically be run with read dir being COADD
# For now I am using test for read and test_2 for write (create data/test_2)

# Names of the data read and write directories 
read = sys.argv[1]
write = sys.argv[2]
dir_read = "data/"+read
dir_write = "data/"+write

# l-cuts
# TODO: do we want this?
lmax = [2000,2000,2600,3200]

# Copied from print_chi2
def obtain_improved_prec(prec,Tmat,prior):
    # The expressions inside the bracket, before the bracket and after the bracket
    bracket = np.linalg.inv((np.dot(np.dot(Tmat,prec),Tmat.T)+prior))
    prebracket = np.dot(prec,Tmat.T)
    postbracket = np.dot(Tmat,prec)
    # Assembling everything into one
    prec_n = prec - np.dot(np.dot(prebracket,bracket),postbracket)
    return prec_n

# Load the data whose precision matrix we would like to modify
s_d = sacc.SACC.loadFromHDF(dir_read+"/power_spectra_wdpj.sacc")

# Get the precision matrix after applying l-cuts
s_d.cullLminLmax([0,0,0,0],lmax)
prec = s_d.precision.getPrecisionMatrix()

# TODO: turn the Tmat computation into a function and ask about HOD params
print ("Loading cached Tmat")
Tmat = np.load("Tmat.npy")

# TODO: get the actual calculation for the prior with CV, noise and smoothing
print("Loading prior with CV, noise and smoothing")
# For now I just saved this from print_chi2
prior_smo = np.load("prior_smo.npy")

# Obtain the new precision matrix
prec_CVnoismo = obtain_improved_prec(prec,Tmat,prior_smo)

# Weirdly it seems like this object actually has the covariance and not precision 
prec_n = sacc.Precision(np.linalg.inv(prec_CVnoismo))

# Create new sacc file
s_n = sacc.SACC(s_d.tracers,s_d.binning,s_d.mean.vector,prec_n,meta=s_d.meta)

# Sanity check of implementation
prec_CVnoismo = s_n.precision.getPrecisionMatrix()

# Degrees of freedom
dof = 84

# log determinant ratio
log_det_prec = np.linalg.slogdet(prec)[1]
log_det_prec_CVnoismo = np.linalg.slogdet(prec_CVnoismo)[1]
ratio_det_CVnoismo = log_det_prec-log_det_prec_CVnoismo
ratio_det_CVnoismo = np.exp(ratio_det_CVnoismo)
print("ratio of det of old prec to CV+noise+smooth prec per dof = ", ratio_det_CVnoismo**(1./dof))
# for the test data, the answer should be 1.1262623661031577 on B.H.'s computer, which it is

# TODO: what about the noise? (perhaps same as COADD)
# Save the new data with the new chain which shall be run via MCMC
s_n.saveToHDF(dir_write+"/power_spectra_wdpj.sacc")

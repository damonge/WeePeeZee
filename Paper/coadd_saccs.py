import sacc
import os
import sys
import numpy as np

# Name of directory where sacc files will be saved
write = sys.argv[1] # I am currently choosing COADDED, NEWCOV_COADDED
dir_write = "data/"+write

# Want to include the new covariance matrix?
want_newcovmat = bool(sys.argv[2])
if want_newcovmat:
    newcov = np.load("/users/boryanah/repos/WeePeeZee/data/NEW_COVMAT/covmat.npy")

# Loading the saccs
sacc_names = ['/users/boryanah/HSC_data/HSC/GAMA15H/CovAna_NoiAna_MskSirius_ClFit_Dpj0_DpjBands1_newDepth/power_spectra_wdpj.sacc', '/users/boryanah/HSC_data/HSC/GAMA09H/CovAna_NoiAna_MskSirius_ClFit_Dpj0_DpjBands1_newDepth/power_spectra_wdpj.sacc', '/users/boryanah/HSC_data/HSC/WIDE12H/CovAna_NoiAna_MskSirius_ClFit_Dpj0_DpjBands1_newDepth/power_spectra_wdpj.sacc', '/users/boryanah/HSC_data/HSC/VVDS/CovAna_NoiAna_MskSirius_ClFit_Dpj0_DpjBands1_newDepth/power_spectra_wdpj.sacc', '/users/boryanah/HSC_data/HSC/XMMLSS/CovAna_NoiAna_MskSirius_ClFit_Dpj0_DpjBands1_newDepth/power_spectra_wdpj.sacc']
saccs = [sacc.SACC.loadFromHDF(fn) for fn in sacc_names]
print("We are coadding "+str(len(sacc_names))+" files.")

# Loading the noise saccs
saccs_noise_names = [os.path.join(os.path.split(fn)[0], 'noi_bias.sacc') for fn in sacc_names]
print('Reading noise saccs {}.'.format(saccs_noise_names))
saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in saccs_noise_names]

# Add precision matrix to noise saccs
for i, s in enumerate(saccs):
    saccs_noise[i].precision = s.precision

# Coadding the files normalizing by area
sacc_coadded = sacc.coadd(saccs, mode='area')
sacc_noise_coadded = sacc.coadd(saccs_noise, mode='area')

# Change the covmat to the new one if requested
if want_newcovmat:
    lmin = [0, 0, 0, 0]
    lmax = [2170.58958919, 2515.39193451, 3185.36076391, 4017.39370804]
    sacc_coadded.cullLminLmax(lmin, lmax)
    sacc_noise_coadded.cullLminLmax(lmin, lmax)
    
    sacc_coadded.precision = sacc.Precision(newcov, 'dense', is_covariance=True)
    sacc_noise_coadded.precision = sacc.Precision(newcov, 'dense', is_covariance=True)

# Recording the final sacc files
if not os.path.exists(dir_write):
    os.makedirs(dir_write)
sacc_coadded.saveToHDF(dir_write+"/power_spectra_wdpj.sacc")
sacc_noise_coadded.saveToHDF(dir_write+"/noi_bias.sacc")

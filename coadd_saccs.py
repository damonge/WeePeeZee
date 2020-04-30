import sacc
import os
import sys

# Name of directory where sacc files will be saved
write = sys.argv[1] # I am currently choosing COADDED
dir_write = "data/"+write

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

# Recording the final sacc files
if not os.path.exists(dir_write):
    os.makedirs(dir_write)
sacc_coadded.saveToHDF(dir_write+"/power_spectra_wdpj.sacc")
sacc_noise_coadded.saveToHDF(dir_write+"/noi_bias.sacc")

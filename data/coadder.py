import sacc
import numpy as np

pz_codes = ['nz_demp', 'nz_ephor', 'nz_ephor_ab', 'nz_frankenz', 'nz_nnpz']
fields = ['GAMA09H', 'GAMA15H', 'HECTOMAP', 'VVDS', 'XMMLSS', 'WIDE12H']

# Read all sacc files
sc = {f:sacc.SACC.loadFromHDF(f+"/power_spectra_wdpj.sacc") for f in fields}
sn = {f:sacc.SACC.loadFromHDF(f+"/noi_bias.sacc") for f in fields}

# Areas
area = np.sum(np.array([sc[f].meta['Area_rad']
                        for f in fields]))
# Spectra
mean = np.sum(np.array([sc[f].mean.vector * sc[f].meta['Area_rad']
                        for f in fields]), axis=0) / area
# Noise
mean_n = np.sum(np.array([sn[f].mean.vector * sc[f].meta['Area_rad']
                          for f in fields]), axis=0) / area
# Covariance
cov = np.sum(np.array([sc[f].precision.getCovarianceMatrix() * sc[f].meta['Area_rad']**2
                       for f in fields]), axis=0) / area**2
# N(z)s
zs = np.array([t.z for t in sc['GAMA09H'].tracers])
Nz = np.sum(np.array([np.array([t.Nz for t in sc[f].tracers]) * sc[f].meta['Area_rad']
                      for f in fields]), axis=0) / area
ec = {pn: np.sum(np.array([np.array([t.extra_cols[pn] for t in sc[f].tracers]) * sc[f].meta['Area_rad']
                           for f in fields]), axis=0) / area for pn in pz_codes}
# Windows
wins = [sacc.Window(sc['GAMA09H'].binning.windows[i_w].ls,
                    np.sum(np.array([sc[f].binning.windows[i_w].w * sc[f].meta['Area_rad']
                                     for f in fields]), axis=0) / area)
        for i_w,ww in enumerate(sc['GAMA09H'].binning.windows)]

# Bins
s_bn = sc['GAMA09H'].binning
s_bn.windows=wins

# Tracers
s_tr = []
for i_t in range(4):
    T = sacc.Tracer('bin_%d'%i_t,
                    'point',
                    zs[i_t], Nz[i_t],
                    exp_sample="HSC_DESC")
    T.addColumns({pn:ec[pn][i_t] for pn in pz_codes})
    s_tr.append(T)

# Signal spectra
s_mean = sacc.MeanVec(mean)
s_prec = sacc.Precision(cov,"dense",is_covariance=True, binning=s_bn)
s_meta = {'Area_rad':area}
s=sacc.SACC(s_tr,s_bn,s_mean,precision=s_prec,meta=s_meta)
s.saveToHDF("COADD/power_spectra_wdpj.sacc")
# Noise spectra
s_mean = sacc.MeanVec(mean_n)
s_bn.windows=None
s=sacc.SACC(s_tr,s_bn,s_mean,meta=s_meta)
s.saveToHDF("COADD/noi_bias.sacc")

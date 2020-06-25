import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import plots,MCSamples
import sys

# names and choices
name_fid = "fid_newcov_april28"
lab_fid = 'Full 14-parameter fit (HSC paper)'#'fiducial, new covmat'
name_marg = "marg_newcov_june19"
lab_marg = 'Marginalized N(z)'#, new covmat'
dir_chains = "/users/boryanah/desclss_chains/"
ups = "_upsample4"

# what are we plotting
HOD_pars = ["lmminp","lmmin","m0p","m0","m1p","m1"]
filename = "triangle_fid_marg"+ups+".png"

def get_par_names(name):
    if 'april29' in name or 'june19' in name:
        n_iter = 12000; w_rat = 6; n_par = 6; b_iter = 2000
        par_names = ["lmminp","lmmin","m0p","m0","m1p","m1"]
        lab_names = ["\\mu_{{\\rm min},p}","\\mu_{\\rm min}","\\mu_{0,p}","\\mu_{0}","\\mu_{1,p}","\\mu_{1}"]
        return n_iter, w_rat, n_par, b_iter, par_names, lab_names

    n_iter = 44000; w_rat = 8; n_par = 14; b_iter = 22000
    par_names = ["zshift_bin0","zshift_bin1","zshift_bin2","zshift_bin3","lmminp","lmmin","m0p","m0","m1p","m1","zwidth_bin0","zwidth_bin1","zwidth_bin2","zwidth_bin3"]
    lab_names = ["\\Delta z_0","\\Delta z_1","\\Delta z_2","\\Delta z_3","\\mu_{{\\rm min},p}","\\mu_{\\rm min}","\\mu_{0,p}","\\mu_{0}","\\mu_{1,p}","\\mu_{1}","z_{w,0}","z_{w,1}","z_{w,2}","z_{w,3}"]
    return n_iter, w_rat, n_par, b_iter, par_names, lab_names


# walkers ratio, number of params and burn in iterations
fiducial_outfile = name_fid+"/"+name_fid+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_fid)
fiducial_chains = np.loadtxt(dir_chains+fiducial_outfile)

# parameters for CV importance sampling
shift_sigmas = np.array([0.008,0.006,0.006,0.007])
width_sigmas = np.array([0.05,0.045,0.04,0.03])

def gaussian(param,sigma):
    return np.exp(-0.5*param**2/sigma**2)

# removing burn-in
#fiducial_chains = fiducial_chains[w_rat*n_par*b_iter:]

# compute weights
weights = np.ones(fiducial_chains.shape[0])
for i in range(4):
    sig = shift_sigmas[i]
    weights[:] *= gaussian(fiducial_chains[:,i],sig)
for i in range(4):
    sig = width_sigmas[i]
    weights[:] *= gaussian(fiducial_chains[:,-4+i],sig)

# feed samples to getdist
fiducial_hsc = MCSamples(samples=fiducial_chains,names=par_names,labels=lab_names,name_tag='Fid',weights=weights)

# load the marginalized dataset
marg_outfile = name_marg+"/"+name_marg+ups+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_marg)
marg_chains = np.loadtxt(dir_chains+marg_outfile)
marg_chains = marg_chains[w_rat*n_par*b_iter:]
print(marg_chains.shape)
marg_hsc = MCSamples(samples=marg_chains,names=par_names,labels=lab_names,name_tag='Marg')

# Triangle plot
g = plots.getSubplotPlotter()
g.triangle_plot([fiducial_hsc,marg_hsc],params=HOD_pars,legend_labels=[lab_fid,lab_marg],filled=True)
plt.savefig(dir_chains+triangle/"+filename)
plt.close()

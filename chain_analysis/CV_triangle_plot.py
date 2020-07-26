import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import plots,MCSamples
import sys
#import plotparams
#plotparams.default()

# names and choices
name_fid = "fid_newcov_april28"
lab_fid = '14-parameter fit (HSC paper)'#'fiducial, new covmat'
name_CV = "CV_newcov_july17"
lab_CV = '14-parameter fit with Cosmic Variance prior'#'Cosmic variance constraint'
name_marg = "marg_newcov_june19"
lab_marg = '6-parameter fit using ${\sf C}_M$'#'Marginalized N(z)'
dir_chains = "/users/boryanah/desclss_chains/"
ups = ''#"_upsample4"

# what are we plotting
HOD_pars = ["lmminp","lmmin","m0p","m0","m1p","m1"]
filename = "triangle_fid_marg"+ups+".pdf"

ranges_dic = {"m1":[12,14],#"lmminp":,
              "lmmin":[11,12.5]} #"m0p":,#"m0":,#"m1p":[],
              

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


# load the fiducial dataset
# walkers ratio, number of params and burn in iterations
fiducial_outfile = name_fid+"/"+name_fid+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_fid)
fiducial_chains = np.loadtxt(dir_chains+fiducial_outfile)

# removing burn-in
fiducial_chains = fiducial_chains[w_rat*n_par*b_iter:]

# load the CV dataset
CV_outfile = name_CV+"/"+name_CV+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_CV)
CV_chains = np.loadtxt(dir_chains+CV_outfile)

# removing burn-in
CV_chains = CV_chains[w_rat*n_par*b_iter:]

# feed samples to getdist
CV_hsc = MCSamples(samples=CV_chains,names=par_names,labels=lab_names,ranges=ranges_dic,name_tag='CV')
fiducial_hsc = MCSamples(samples=fiducial_chains,names=par_names,labels=lab_names,ranges=ranges_dic,name_tag='Fid')

# load the marginalized dataset
marg_outfile = name_marg+"/"+name_marg+ups+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_marg)
marg_chains = np.loadtxt(dir_chains+marg_outfile)

# removing burn-in
marg_chains = marg_chains[w_rat*n_par*b_iter:]

# feed samples to getdist
marg_hsc = MCSamples(samples=marg_chains,names=par_names,labels=lab_names,ranges=ranges_dic,name_tag='Marg')

def print_bounds(ms):
 
    for i in range(len(par_names)):
        string_all = ""
        for j in range(len(ms)):
            m = ms[j]
            f1 = m.getLatex(par_names,limit=1)
            p, v1 = f1
            f2 = m.getLatex(par_names,limit=2)
            p, v2 = f2
            value = v2[i]
            value = '^'+value.split('^')[-1]
            string_all += "$"+v1[i]+"$, $"+value+"$"
            if j != len(ms)-1:
                string_all += " & "
        print("$"+p[i]+"$ & "+string_all+" \\\\ [1ex]")
        
print("\\begin{table}")
print("\\begin{center}")
print("\\begin{tabular}{c | c c c c} ")
print(" \\hline\\hline")
print(" Parameter & HSC cov. [68\%, 95\%] & CV constraints [68\%, 95\%] & Marg. $N(z)$ [68\%, 95\%] \\\\ [0.5ex] ")
print(" \\hline")
print(" $\\chi^2/\\nu$ & 87.49/80 & 88.29/80 & 88.54/82.32 \\\\ ")
margs = [fiducial_hsc, CV_hsc, marg_hsc]
print_bounds(margs)
print(" \\hline")
print(" \\hline")
print("\\end{tabular}")
print("\\end{center}")
print("\\label{tab:chi2_tests}")
print("\\caption{Table}")
print("\\end{table}")


# Triangle plot
g = plots.getSubplotPlotter()
g.settings.legend_fontsize = 20
g.settings.scaling_factor = 0.1
g.triangle_plot([fiducial_hsc,marg_hsc,CV_hsc],params=HOD_pars,legend_labels=[lab_fid,lab_marg,lab_CV],filled=True)
plt.savefig("../Paper/"+filename)
plt.close()

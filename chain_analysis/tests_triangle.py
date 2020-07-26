import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import plots,MCSamples
import sys

hod_dic = {'lmmin':11.88, 'lmminp':-0.5,
              'm0':5.7, 'm0p':2.5,
              'm1':13.08, 'm1p':0.9,}
              
new_hod_dic = {'lmmin': 12.1,'lmminp': -2.5,
               'm0': 9.7, 'm0p': -2.5,
               'm1': 13.35, 'm1p': -1.7}

# parameters
n_iter = 12000; w_rat = 6; n_par = 6; b_iter = 2000
par_names = ["lmminp","lmmin","m0p","m0","m1p","m1"]
lab_names = ["\\mu_{{\\rm min},p}","\\mu_{\\rm min}","\\mu_{0,p}","\\mu_{0}","\\mu_{1,p}","\\mu_{1}"]

# what are we plotting
HOD_pars = ["lmminp","lmmin","m0p","m0","m1p","m1"]
filename = "triangle_marg_tests.pdf"#.png"
dir_chains = "/users/boryanah/desclss_chains/"

# names and choices
name_marg = "marg_newcov_june19"#"marg_newcov_april29"
name_test1 = "marg_newcov_1sigma_june21"#"marg_newcov_1sigma_may1"
name_test2 = "marg_newcov_nosmooth_june22"#"marg_newcov_nosmooth_may2"
name_test3 = "marg_newcov_largenoise_june23"#"marg_newcov_largenoise_may3"

lab_marg = '6-parameter fit using ${\sf C}_M$, fiducial'#'Marginalized N(z), fiducial'#'marg Nz, new covmat'
lab_test1 = '6-parameter fit using ${\sf C}_M$, $1\sigma$ off'#'Marginalized N(z), $1\sigma$ off'#'marg Nz 1 sigma, new covmat'
lab_test2 = '6-parameter fit using ${\sf C}_M$, no smoothing'#'Marginalized N(z), no smoothing'#'marg Nz no smoothing, new covmat'
lab_test3 = '6-parameter fit using ${\sf C}_M$, large diagonal noise'#'Marginalized N(z), large diagonal noise'#'marg Nz large noise, new covmat'

# walkers ratio, number of params and burn in iterations
marg_outfile = name_marg+"/"+name_marg+".txt"
marg_test1_outfile = name_test1+"/"+name_test1+".txt"
marg_test2_outfile = name_test2+"/"+name_test2+".txt"
marg_test3_outfile = name_test3+"/"+name_test3+".txt"

def get_samples(outfile):
    marg_chains = np.loadtxt(dir_chains+outfile)
    marg_chains = marg_chains[w_rat*n_par*b_iter:]
    print(marg_chains.shape)
    hsc = MCSamples(samples=marg_chains,names=par_names,labels=lab_names,name_tag='Marg')
    return hsc

marg_hsc = get_samples(marg_outfile)
marg_test1 = get_samples(marg_test1_outfile)
marg_test2 = get_samples(marg_test2_outfile)
marg_test3 = get_samples(marg_test3_outfile)

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
print(" Parameter & Fiducial [68\%, 95\%] & Test 1 [68\%, 95\%] & Test 2 [68\%, 95\%] & Test 3 [68\%, 95\%] \\\\ [0.5ex] ")
print(" \\hline")
print(" $\\chi^2/\\nu$ & 88.54/88 & 87.67/88 & 88.56/88 & 78.61/88 \\\\ ")
margs = [marg_hsc, marg_test1, marg_test2, marg_test3]
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
g.triangle_plot([marg_test3,marg_test1,marg_test2,marg_hsc],params=HOD_pars,legend_labels=[lab_test3,lab_test1,lab_test2,lab_marg],filled=True)
lw = 1.5
for i,key_i in enumerate(hod_dic.keys()):
    for j,key_j in enumerate(hod_dic.keys()):
        if j == i:
            print("i,j = ",i,j)
            ax = g.get_axes_for_params(key_i)
            print(ax)
            g.add_x_marker(marker=hod_dic[key_i],color='blue',ax=ax,lw=lw,ls='--')
            g.add_x_marker(marker=new_hod_dic[key_i],color='gray',ax=ax,lw=lw,ls='--')
            ax = None
        else:#if j > i:
            print("i,j = ",i,j)
            ax = g.get_axes((key_j,key_i))
            if ax is not None:
                ax.scatter(np.array(hod_dic[key_j]),np.array(hod_dic[key_i]),color='blue',s=18.,marker='x')
                ax.scatter(np.array(new_hod_dic[key_j]),np.array(new_hod_dic[key_i]),color='gray',s=18.,marker='x')
            print(ax)
            #g.add_y_marker(marker=hod_dic[key_i],color='blue',ax=ax,lw=lw,ls='--')
            #g.add_y_marker(marker=new_hod_dic[key_i],color='gray',ax=ax,lw=lw,ls='--')
            #g.add_x_marker(marker=hod_dic[key_j],color='blue',ax=ax,lw=lw,ls='--')
            #g.add_x_marker(marker=new_hod_dic[key_j],color='gray',ax=ax,lw=lw,ls='--')
            ax = None
plt.savefig("/users/boryanah//repos/WeePeeZee/Paper/"+filename)
plt.close()

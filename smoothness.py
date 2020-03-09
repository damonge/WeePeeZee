import numpy as np
# Code copied from NzGCWL, but didn't want to make the two repos dep on each o

def obtain_smoothing_D(N_tracers,N_zs,Delta_z,first,second,sum,corr=0.1):
    # first and second derivatives
    D0 = np.zeros((N_zs*N_tracers,N_zs*N_tracers))
    D1 = np.zeros((N_zs*N_tracers,N_zs*N_tracers))
    D2 = np.zeros((N_zs*N_tracers,N_zs*N_tracers))
    
    for i in range(N_tracers):
        D0[i*N_zs:(i+1)*N_zs,i*N_zs:(i+1)*N_zs] += np.ones((N_zs,N_zs))
        for j in range (0,N_zs-1):
            lam = np.zeros(N_zs*N_tracers)
            lam[i*N_zs+j]=1
            lam[i*N_zs+j+1]=-1
            D1+=np.outer(lam,lam)
        for j in range (1,N_zs-1):
            lam = np.zeros(N_zs*N_tracers)
            lam[i*N_zs+j-1]=-1
            lam[i*N_zs+j]=2
            lam[i*N_zs+j+1]=-1
            D2+=np.outer(lam,lam)
            

    S = Delta_z
    s = corr

    sigma1 = (Delta_z**2/(10.*S*s))
    sigma2 = (Delta_z**4/(10.*S*s**2))
    sigma1sq = sigma1**2
    sigma2sq = sigma2**2

    #TODO: ask Anze about this normalization and about Delta_z 
    D0 /= N_tracers**2
    D1 /= (sigma1sq)
    D2 /= (sigma2sq)

    if first == False: D1 *= 0
    if second == False: D2 *= 0
    if sum == False: D0 *= 0

    D = D0+D1+D2
    return D

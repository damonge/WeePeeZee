import numpy as np
# Code copied from NzGCWL, but didn't want to make the two repos dep on each o

def obtain_smoothing_D(sacc,first,second):
    # first and second derivatives
    N_tracers=len(sacc.tracers) # num tracers
    N_zs=len(sacc.tracers[0].z) # zbins per tracer

    D1all = np.zeros((N_zs*N_tracers,N_zs*N_tracers))
    D2all = np.zeros((N_zs*N_tracers,N_zs*N_tracers))


    for i in range(N_tracers):
        D1=np.zeros((N_zs,N_zs))
        D2=np.zeros((N_zs,N_zs))
        afd,asd=[],[]
        for j in range (0,N_zs-1):
            lam = np.zeros(N_zs)
            lam[j]=1
            lam[j+1]=-1
            D1+=np.outer(lam,lam)
            afd.append(np.dot(lam,sacc.tracers[i].Nz))
        for j in range (1,N_zs-1):
            lam = np.zeros(N_zs)
            lam[j-1]=-1
            lam[j]=2
            lam[j+1]=-1
            D2+=np.outer(lam,lam)
            asd.append(np.dot(lam,sacc.tracers[i].Nz))
            
        afd=np.array(afd).var()
        asd=np.array(asd).var()
        D1all[i*N_zs:(i+1)*N_zs,i*N_zs:(i+1)*N_zs]=D1/afd
        D2all[i*N_zs:(i+1)*N_zs,i*N_zs:(i+1)*N_zs]=D2/asd
        
    if first == False: D1all *= 0
    if second == False: D2all *= 0

    D = D1all+D2all
    return D

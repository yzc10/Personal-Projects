# Segment video into separate videos of the background and the moving object(s)

import os
import pandas as pd
import numpy as np
import scipy as sc
import dask.array as da
import matplotlib.pyplot as plt
import time

def gen_matrices(m,n):
    np.random.seed(1)
    M = pd.read_csv('BasketballPlayer.csv',header=None).to_numpy()
    Z = np.zeros(shape=(m,n))
    L = np.zeros(shape=(m,n))
    S = np.zeros(shape=(m,n))
    return L,S,M,Z

def compute_r(Lk1,Lk0,Sk1,Sk0):
    L = np.linalg.norm(Lk1-Lk0)/(1+np.linalg.norm(Lk1))
    S = np.linalg.norm(Sk1-Sk0)/(1+np.linalg.norm(Sk1))
    return max(L,S)

def solve_admm(m,n,filename,k=10):
    start = time.perf_counter()
    L,S,M,Z = gen_matrices(m,n)
    Lk1,Lk0 = L,L
    Sk1,Sk0 = S,S
    sig = 1/400
    tau = 1.618
    lbda = (max(m,n))**-0.5
    rho = 1.3
    k = 0
    score = 1
    df_results = pd.DataFrame(columns=['rk','running time'])
    
    while score >= 10**-4 and k <= 200:
        sig_inv = 1/sig
        T = M - S - sig_inv*Z
        svd_T = sc.linalg.svd(T,full_matrices=False,overwrite_a=True,check_finite=False)
        U = svd_T[0]
        d = svd_T[1]
        Vt = svd_T[2]
        g = np.maximum(d-sig_inv,0)
        L = U @ (np.diag(g) @ Vt)
        X = M - L - sig_inv*Z
        S = np.sign(X)*np.maximum(np.abs(X)-lbda*sig_inv,0)
        Z += tau * sig * (L + S - M)
        
        sig *= rho
        Lk0,Lk1 = Lk1,L
        Sk0,Sk1 = Sk1,S
        score = compute_r(Lk1,Lk0,Sk1,Sk0)
        df_results.loc[k,'rk'] = score
        df_results.loc[k,'running time'] = time.perf_counter() - start
        k += 1
    print(k,time.perf_counter()-start,score)
    df_results.to_csv(filename,index=False)
    
    L = pd.DataFrame(L)
    S = pd.DataFrame(S)
    L.to_csv('q4b_L.csv',index=False)
    S.to_csv('q4b_S.csv',index=False)
    L = L.to_numpy()
    S = S.to_numpy()
    svd_L = np.linalg.svd(L,full_matrices=False,compute_uv=False)
    rank_L = (svd_L>10**-6).sum()
    nonzero_S = (S.flatten()>10**-6).sum()
    print('rank of L = ',rank_L,'\nno. of nonzero elements in S = ',nonzero_S)

def read_MLS():
    M = pd.read_csv('BasketballPlayer.csv',header=None).to_numpy()
    L = pd.read_csv('q4b_L.csv').to_numpy()
    S = pd.read_csv('q4b_S.csv').to_numpy()
    L = (L-L.min())/(L.max()-L.min())
    S = (S-S.min())/(S.max()-S.min())
    return M,L,S

def gen_frame(n): # frame number
    M,L,S = read_MLS()
    m = M[:,n]
    l = L[:,n]
    s = S[:,n]
    m = m.reshape(1374,918).T
    l = l.reshape(1374,918).T
    s = s.reshape(1374,918).T
    n = 1
    for i in (m,l,s):
        plt.subplot(1,3,n)
        plt.imshow(i)
        plt.xticks([])
        plt.yticks([])
        n += 1
    plt.show()

def gen_video(n):
    start = time.perf_counter()
    M,L,S = read_MLS()
    for i in range(n):
        plt.imshow(L[:,i].reshape(1374,918).T)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('q4d\L_{}.png'.format(str(i+1)))

        plt.imshow(S[:,i].reshape(1374,918).T)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('q4d\S_{}.png'.format(str(i+1)))

if __name__ == '__main__':
    path = r'{PATH}'
    os.chdir(path)
    
    # Solve for 
    solve_admm(m=1261332,n=112,filename='robust_pca_results.csv')

    # Visualize 20th frame
    gen_frame(n=19) 

    # Export images for video creation
    gen_video(n=112) 
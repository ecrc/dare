import numpy as np
from astropy.io.fits import open as fitsopen
from astropy.io.fits import writeto as fitswrite
import sys
from time import time
import os
from os.path import join
from scipy.linalg import lu_factor, lu_solve,solve
# running cupy in vscode jupyter is very complicated because of cuda configuration.
# we move it to pure py file and execute in shell.
import cupy as cp 
from cupyx.scipy.linalg import lu_factor as cplu_factor
from cupyx.scipy.linalg import lu_solve as cplu_solve


def fitsread(filename):
    return (fitsopen(filename)[0].data)

dim = 7090
prefix = "/home/hongy0a/data/astronomydare/" + str(dim)
At = fitsread(join(prefix,"At{}_transpose.fits".format(dim))).T;print(At.shape)
Bt = fitsread(join(prefix,"Btinitial{}_transpose.fits".format(dim))).T; print(Bt.shape)
Qt = fitsread(join(prefix,"Qt{}.fits".format(dim))); print(Qt.shape)
Rt = fitsread(join(prefix,"Rt{}.fits".format(dim))); print(Rt.shape)
BinvRt = fitsread(join(prefix,"BinvRt{}_transpose.fits".format(dim))).T

def verify_DARE(A,B,Q,R,X):
    LHS = X
    RHS = A.T @ X @ A - (A.T @ X @ B) @ cp.linalg.solve(R + B.T @ X @ B, B.T @ X @ A) + Q
    print("fro_norm(LHS - RHS)=",cp.linalg.norm(LHS-RHS,ord="fro"))

def DAREgpu_ccode(A,B,BinvR,Q,R,epochs=50):
    A = cp.asarray(A)
    B = cp.asarray(B)
    BinvR = cp.asarray(BinvR)
    Q = cp.asarray(Q)
    R = cp.asarray(R)
    Id = cp.eye(Q.shape[0])
    alpha   = cp.copy(A)
    betaold    = cp.copy(BinvR@BinvR.T)
    gamma   = cp.copy(Q)
    start = time()
    for it in range(1,epochs):
        beta = betaold.copy()
        common  = Id + beta@gamma
        wrong = fitsread("/home/hongy0a/research/dare-dev/build/common_transpose{}.fits".format(it)).T
        print(" common close ", cp.allclose(common, wrong))
        alphaold = alpha.copy()
        wrong = fitsread("/home/hongy0a/research/dare-dev/build/alphaold_transpose{}.fits".format(it)).T
        print(" alphaold close ", cp.allclose(alphaold, wrong))
        lu,piv = cplu_factor(common)
        alpha = cplu_solve((lu,piv), alpha, overwrite_b=True)
        wrong = fitsread("/home/hongy0a/research/dare-dev/build/alpha_transpose{}.fits".format(it)).T
        print(" alpha close ", cp.allclose(alpha, wrong))
        buf1 = alphaold.T @ gamma
        gamma += buf1 @ alpha
        wrong = fitsread("/home/hongy0a/research/dare-dev/build/gamma_transpose{}.fits".format(it)).T
        print(" gamma close ", cp.allclose(gamma, wrong))
        diff = cp.linalg.norm(buf1,ord='fro')
        print("it {} difference: {:.6e} tol 1e-15".format(it, cp.asnumpy(diff)))
        if (diff<1e-15):
            break
        beta = cplu_solve((lu,piv), beta, overwrite_b=True)
        wrong = fitsread("/home/hongy0a/research/dare-dev/build/beta_transpose{}.fits".format(it)).T
        print(" beta close ", cp.allclose(beta, wrong))
        buf1 = beta @ alphaold.T
        betaold += alphaold @ buf1
        buf1 = alphaold @ alpha
        alpha = buf1.copy()
    end = time()
    print("end-start",end-start)
    lu,piv = cplu_factor(R+B.T@gamma@B)
    tmp = cplu_solve((lu,piv),B.T)
    res = tmp@gamma@A
    return cp.asnumpy(res)


def DAREgpu(A,B,BinvR,Q,R,epochs=50):
    A = cp.asarray(A)
    B = cp.asarray(B)
    BinvR = cp.asarray(BinvR)
    Q = cp.asarray(Q)
    R = cp.asarray(R)
    Id = cp.eye(Q.shape[0])
    alpha   = cp.copy(A)
    beta    = cp.copy(BinvR@BinvR.T)
    gamma   = cp.copy(Q)
    start = time()
    for it in range(epochs):
        diff = cp.linalg.norm(alpha.T@gamma,ord='fro')
        print("it {} difference: {:.6e} tol 1e-16".format(it, cp.asnumpy(diff)))
        if (diff<1e-16):
            break
        common  = Id + beta@gamma
        lu,piv = cplu_factor(common)
        alphasol = cplu_solve((lu,piv), alpha)
        print(cp.linalg.norm(common @ alphasol - alpha))
        betasol = cplu_solve((lu,piv), beta)
        wrong = cp.load("wrong.npy");print(" beta close ", cp.allclose(betasol, wrong))
        print(cp.linalg.norm(common @ betasol - beta))
        gamma += alpha.T @ gamma @ alphasol
        beta  += alpha @ betasol @alpha.T
        alphanew   = alpha@alphasol
        alpha = alphanew.copy()
        break
    return
    end = time()
    print("end-start",end-start)
    lu,piv = cplu_factor(R+B.T@gamma@B)
    tmp = cplu_solve((lu,piv),B.T)
    res = tmp@gamma@A
    return cp.asnumpy(res)

# DAREgpu_ccode(At,Bt,BinvRt,Qt,Rt)
# DAREgpu(At,Bt,BinvRt,Qt,Rt)


def verify_DARE2(A,B,Q,R,X):
    LHS = X
    RHS = A.T @ X @ A - (A @ X @ B) @ cp.linalg.solve(R + B.T @ X @ B, B.T @ X @ A.T) + Q
    print("fro_norm(LHS - RHS)=",cp.linalg.norm(LHS-RHS,ord="fro"))


def DAREgpu2(A,B,Q,R,epochs=50):
    A = cp.asarray(A).astype(cp.float32)
    B = cp.asarray(B).astype(cp.float32)
    Q = cp.asarray(Q).astype(cp.float32)
    R = cp.asarray(R).astype(cp.float32)
    Id = cp.eye(Q.shape[0]).astype(cp.float32)
    # A = cp.asarray(A)
    # B = cp.asarray(B)
    # Q = cp.asarray(Q)
    # R = cp.asarray(R)
    # Id = cp.eye(Q.shape[0])
    alpha   = cp.copy(A)
    beta    = cp.copy(B@cp.linalg.solve(R,B.T))
    gamma   = cp.copy(Q)
    start = time()
    for it in range(epochs):
        common  = Id + beta@gamma
        q,r = cp.linalg.qr(common)
        p = cp.dot(q.T, alpha)
        alphasol = cp.dot(cp.linalg.inv(r), p)
        p = cp.dot(q.T, beta)
        betasol = cp.dot(cp.linalg.inv(r), p)
        # lu,piv = cplu_factor(common)
        # alphasol = cplu_solve((lu,piv), alpha)
        # betasol = cplu_solve((lu,piv), beta)
        alphanew   = alpha@betasol@alpha.T
        beta += alphanew
        gammadelta = alpha.T @ gamma @ alphasol
        gamma += gammadelta
        print("%d/%d Complete" % (it+1, epochs))
        diff = cp.max(cp.abs(gammadelta.flatten())) / cp.max(cp.abs(gamma.flatten()))
        print("difference: %f" % diff)
        if (diff<1e-16):
            break
        alpha = cp.copy(alphanew)
        verify_DARE2(A,B,Q,R,gamma)
    end = time()
    print("end-start",end-start)
    res = gpusolve(R+B.T@gamma@B,B.T)@gamma@A
    return cp.asnumpy(res)


# X = ATXA - ATXB()-1BTXA + Q
def verify_DARE3(A,B,Q,R,X):
    LHS = X
    RHS = A.T @ X @ A - (A.T @ X @ B) @ cp.linalg.solve(R + B.T @ X @ B, B.T @ X @ A) + Q
    print("fro_norm(LHS - RHS)=",cp.linalg.norm(LHS-RHS,ord="fro"))


def DAREgpu3(A,B,Q,R,epochs=50):
    A = cp.asarray(A)
    B = cp.asarray(B)
    Q = cp.asarray(Q)
    R = cp.asarray(R)
    Id = cp.eye(Q.shape[0])
    alpha   = cp.copy(A)
    beta    = cp.copy(B@cp.linalg.solve(R,B.T))
    gamma   = cp.copy(Q)
    start = time()
    for it in range(epochs):
        common  = Id + beta@gamma
        lu,piv = cplu_factor(common)
        alphasol = cplu_solve((lu,piv), alpha)
        betasol = cplu_solve((lu,piv), beta.T)
        alphanew = alpha @ alphasol
        beta += alpha @ betasol.T @ alpha.T
        gammadelta = alphasol.T @ gamma @ alpha
        gamma += gammadelta
        print("%d/%d Complete" % (it+1, epochs))
        diff = cp.max(cp.abs(gammadelta.flatten())) / cp.max(cp.abs(gamma.flatten()))
        print("difference: %f" % diff)
        if (diff<1e-7):
            break
        alpha = alphanew.copy()
        verify_DARE3(A,B,Q,R,gamma)
    end = time()
    print("end-start",end-start)
    res = gpusolve(R+B.T@gamma@B,B.T)@gamma@A
    return cp.asnumpy(res)



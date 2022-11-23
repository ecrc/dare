// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include "dplasma.h"
#include "parsecsrc/parsecio.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include "parsecsrc/util.hpp"
#include "parsecsrc/flops.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include <vector>
#include <chrono>
#include <stdio.h>
using namespace std;
using namespace Eigen;
/**
 * @brief Get the problem dimensions from the input files and check consistency
 *
 * @param datapath : (string) : path to the data folder
 * @param file_At  : (FitsFile&) : handle for the fits file of the matrix A.T
 * @param file_BinvRt : (FitsFile&) : handle for the fits file of the matrix inv(B)*R.T
 * @param file_Btinitial : (FitsFile&) : handle for the fits file of the matrix Binitial.T
 * @param file_Qt : (FitsFile&) : handle for the fits file of the matrix Q.T
 * @param file_Rt : (FitsFile&) : handle for the fits file of the matrix R.T
 * @param N : (int&) : (out) first dimension of the problem
 * @param NINSTR : (int&) : (out) second dimension of the problem
 */
void get_matrices_dimensions(const std::string &datapath, FitsFile &file_At, FitsFile &file_BinvRt,
                             FitsFile &file_Btinitial, FitsFile &file_Qt, FitsFile &file_Rt, int &N, int &NINSTR)
{
    std::vector<long> shape_Btinitial = file_Btinitial.get_data_shape(0);
    N = shape_Btinitial[0];
    NINSTR = shape_Btinitial[1];
    cout << N << ", " << NINSTR << endl;
    // checking dimensions compatibility
    int error = 0;
    if (file_At.get_data_shape(0) != std::vector<long>{N, N})
    {
        error = 1;
        std::cerr << "Dimensions of At are not good" << std::endl;
    }
    if (file_Btinitial.get_data_shape(0) != std::vector<long>{N, NINSTR})
    {
        error = 1;
        std::cerr << "Dimensions of Binitial are not good" << std::endl;
    }
    if (file_Qt.get_data_shape(0) != std::vector<long>{N, N})
    {
        error = 1;
        std::cerr << "Dimensions of Qt are not good" << std::endl;
    }
    if (file_Rt.get_data_shape(0) != std::vector<long>{NINSTR, NINSTR})
    {
        error = 1;
        std::cerr << "Dimensions of Rt are not good" << std::endl;
    }
    if (error)
    {
        throw std::runtime_error("Matrices dimensions inconsistent");
    }
}

#define REGDENSE(var, M, N)                                                                       \
    parsec_matrix_block_cyclic_t var;                                                             \
    parsec_matrix_block_cyclic_init(&var, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,               \
                                    0, M, N, M, N, 0, 0,                                          \
                                    M, N, P, world / P, 1, 1, 0, 0);                              \
    var.mat = malloc((size_t)var.super.nb_local_tiles * (size_t)var.super.bsiz * sizeof(double)); \
    dplasma_dplrnt(parsec, 0, &var.super, random_seed++);

#define REGVAR(var, M, N)                                                                         \
    parsec_matrix_block_cyclic_t var;                                                             \
    parsec_matrix_block_cyclic_init(&var, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,               \
                                    0, mb, nb, M, N, 0, 0,                                        \
                                    M, N, P, world / P, 1, 1, 0, 0);                              \
    var.mat = malloc((size_t)var.super.nb_local_tiles * (size_t)var.super.bsiz * sizeof(double)); \
    dplasma_dplrnt(parsec, 0, &var.super, random_seed++);

#define SUPER(x) &x.super

#define ADD(handle, event) parsec_context_add_taskpool(handle, event);

#define START(handle) parsec_context_add_taskpool(handle);

#define BARRIER(handle) parsec_context_add_taskpool(handle);


#define STEP(handle, event)                     \
    parsec_context_add_taskpool(handle, event); \
    if (sync)                                   \
    {                                           \
        parsec_context_start(handle);           \
        parsec_context_wait(handle);            \
    }

template <typename T>
void remapdata(parsec_matrix_block_cyclic_t *src, parsec_matrix_block_cyclic_t *dest)
{
    int M = dest->super.m;
    int N = dest->super.n;
    int mb = dest->super.mb;
    int nb = dest->super.nb;
    int mbt, nbt;
    mbt = M / mb;
    if (mbt * mb != M)
        mbt += 1;
    nbt = N / nb;
    if (nbt * nb != N)
        nbt += 1;
    T *srcptr = reinterpret_cast<T *>(src->mat);
    T *destptr = reinterpret_cast<T *>(dest->mat);
    REP(j, nbt)
    REP(i, mbt)
    {
        REP(col, nb)
        REP(row, mb)
        {
            long gx, gy;
            gx = j * nb + col;
            gy = i * mb + row;
            if (gx >= N || gy >= M)
                continue;
            long long gcnt = (j * nb + col) * M + i * mb + row;
            long long tcnt = (j * mbt + i) * mb * nb + col * mb + row;
            destptr[tcnt] = srcptr[gcnt];
        }
    }
}

template <typename T>
void convert2dense(parsec_matrix_block_cyclic_t *src, parsec_matrix_block_cyclic_t *dest,
                   int M, int N, int mb, int nb)
{
    int mbt, nbt;
    mbt = M / mb;
    if (mbt * mb != M)
        mbt += 1;
    nbt = N / nb;
    if (nbt * nb != N)
        nbt += 1;
    T *srcptr = reinterpret_cast<T *>(src->mat);
    T *destptr = reinterpret_cast<T *>(dest->mat);
    REP(j, nbt)
    REP(i, mbt)
    {
        REP(col, nb)
        REP(row, mb)
        {
            long gx, gy;
            gx = j * nb + col;
            gy = i * mb + row;
            if (gx >= N || gy >= M)
                continue;
            long long gcnt = (j * nb + col) * M + i * mb + row;
            long long tcnt = (j * mbt + i) * mb * nb + col * mb + row;
            srcptr[gcnt] = destptr[tcnt];
        }
    }
}

void savematrix(parsec_matrix_block_cyclic_t *tilemat, string fileprefix)
{
    int m = tilemat->super.m;
    int n = tilemat->super.n;
    int mb = tilemat->super.mb;
    int nb = tilemat->super.nb;
    parsec_matrix_block_cyclic_t var;
    parsec_matrix_block_cyclic_init(&var, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                    0, m, n, m, n, 0, 0,
                                    m, n, 1, 1, 1, 1, 0, 0);
    var.mat = malloc((size_t)var.super.nb_local_tiles * (size_t)var.super.bsiz * sizeof(double));
    convert2dense<double>(&var, tilemat, m, n, mb, nb);
    write_desc<double>(fileprefix, &var);
    // vector<double> v(m * n);
    // memcpy(v.data(), var.mat, sizeof(double) * m * n);
    // cnpy::npy_save(fileprefix + "_plasma.npy", v);
}

// void readmatrix(parsec_matrix_block_cyclic_t *tilemat, string filename)
// {
//     long long int elems = tilemat->super.m * tilemat->super.n;
//     vector<double> buffer(elems, 0.0);
//     auto array = cnpy::npy_load(filename);
//     memcpy((void*)tilemat->mat, (void*)array.data<double>(), sizeof(double)*elems);
// }

#define CHECKINFO(info) if(info!=0) printf("LINE %d error!",__LINE__);  
#define ENQUEUE(task) parsec_enqueue(parsec, task);
int main(int argc, char **argv)
{
    parsec_context_t *parsec = NULL;
    int rank, world;
    int mb, nb;
    mb = nb = 1024;
    int ib = 128;
    int cores = 1;
    size_t i, j;
    int it = 0;
    double norm = -1;
    double tol = 1e-15;
    int N;                                          // matrix order
    int NINSTR;                                     // number of RHS vectors
    int NCPUS;                                      // number of cores to use
    int NGPUS;                                      // number of gpus (cuda devices) to use
    int P = 1, Q;                                   // MPI process grid
    int seedA = 1, seedB = 2, seedR = 3;            // seed for random number generation
    int seedalpha = 4, seedbeta = 5, seedgamma = 6; // seed for random number generation
    int info;
    int trace = 0;

    FitsFile file_At;
    FitsFile file_BinvRt;
    FitsFile file_Btinitial;
    FitsFile file_Qt;
    FitsFile file_Rt;
    // ArgsParser argparser(argc, argv);
    // std::string datapath = argparser.getstring("datapath");
    //string datapath = "/home/hongy0a/data/astronomydare/7090";
    string datapath = ""; // currently use random dataset
    N = 7090;
    NINSTR = 18624;
    if(datapath!= ""){
        // file_At.set(datapath+"/At"+to_string(N)+"_transpose.fits","r");
        // file_BinvRt.set(datapath+"/BinvRt"+to_string(N)+"_transpose.fits","r");
        // file_Btinitial.set(datapath+"/Btinitial"+to_string(N)+"_transpose.fits","r");
        file_At.set(datapath+"/At.fits","r");
        file_BinvRt.set(datapath+"/BinvRt.fits","r");
        file_Btinitial.set(datapath+"/Btinitial.fits","r");
        file_Qt.set(datapath+"/Qt.fits","r");
        file_Rt.set(datapath+"/Rt.fits","r");
        get_matrices_dimensions(datapath,file_At,file_BinvRt,file_Btinitial,file_Qt,file_Rt, N, NINSTR);
    }

#ifdef PARSEC_HAVE_MPI
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    Q = world / P;
    printf("P %d Q %d rank %d \n", P, Q, rank);
    cores = -1;
    parsec = parsec_init(cores, &argc, &argv);

    int random_seed = 0;
    // load At
    REGDENSE(Atdense, N, N);
    cout << "load At" << endl;
//    read_desc<double>(file_At, &Atdense);
    REGVAR(At, N, N);
    remapdata<double>(&Atdense, &At);

    // load Bt
    REGDENSE(Btdense, N, NINSTR);
    cout << "load Btinit" << endl;
//    read_desc<double>(file_Btinitial, &Btdense);
    REGVAR(Bt, N, NINSTR);
    remapdata<double>(&Btdense, &Bt);

    REGDENSE(BinvRtdense, N, NINSTR);
    cout << "load BinvR" << endl;
//    read_desc<double>(file_BinvRt, &BinvRtdense);
    REGVAR(BinvRt, N, NINSTR);
    remapdata<double>(&BinvRtdense, &BinvRt);

    REGDENSE(Qtdense, N, N);
    cout << "load Q" << endl;
//    read_desc<double>(file_Qt, &Qtdense);
    REGVAR(Qt, N, N);
    remapdata<double>(&Qtdense, &Qt);

    REGDENSE(Rtdense, NINSTR, NINSTR);
    cout << "load R" << endl;
//    read_desc<double>(file_Rt, &Rtdense);
    REGVAR(Rt, NINSTR, NINSTR);
    remapdata<double>(&Rtdense, &Rt);   
    REGDENSE(Koutdense, NINSTR, N);
    REGVAR(alpha, N, N);
    REGVAR(alphaold, N, N);
    REGVAR(beta, N, N);
    REGVAR(betaold, N, N);
    REGVAR(gamma, N, N);
    REGVAR(ws1, N, N);
    REGVAR(ws2, N, N);
    REGVAR(ws3, N, N);
    REGVAR(ws4, N, N);
    REGVAR(ws5, N, N);
    REGVAR(ws6, N, N);
    REGVAR(ws7, NINSTR, N);
    REGVAR(Kout, NINSTR, N);
    REGVAR(gammaold, N, N);
    REGVAR(common, N, N);
    REGVAR(Id, N, N);
    int sync = 1;
    int correctness = 0;
    double flops = 0;
    auto start = std::chrono::steady_clock::now();
    dplasma_dlacpy(parsec, dplasmaUpperLower, SUPER(At), SUPER(alpha));
    dplasma_dlacpy(parsec, dplasmaUpperLower, SUPER(Qt), SUPER(gamma));
    
    dplasma_dgemm(parsec, dplasmaNoTrans, dplasmaTrans, 1.0, SUPER(BinvRt), SUPER(BinvRt), 0.0, SUPER(beta));
    flops += FLOPS_DGEMM(N, N, NINSTR);
    // set Id
    dplasma_dlaset(parsec,dplasmaUpperLower, 0.0, 1.0, SUPER(Id));
    int maxit = 20;
    
    for (auto iter = 0; iter < maxit; iter++)
    {
        
        // backup beta and alpha
        dplasma_dlacpy(parsec,dplasmaUpperLower, SUPER(beta), SUPER(betaold));
        dplasma_dlacpy(parsec,dplasmaUpperLower, SUPER(alpha), SUPER(alphaold));
        // common  = Id + beta@gamma | beta@gamma => common, common += Id
        dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(beta), SUPER(gamma), 0.0, SUPER(common));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dgeadd(parsec,dplasmaNoTrans, 1.0, SUPER(Id), 1.0, SUPER(common));
        // savematrix(&common, "betaxgamma");
        // trf_nopiv common | common => Lower Upper Triangular matrix
        dplasma_dgetrf_nopiv(parsec,SUPER(common));
        // savematrix(&common, "trfnopiv");
        // solve(common,alpha) | alpha => common^{-1} x alpha
        dplasma_dtrsm(parsec,dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaUnit, 1.0, SUPER(common), SUPER(alpha));
        dplasma_dtrsm(parsec,dplasmaLeft, dplasmaUpper, dplasmaNoTrans, dplasmaNonUnit, 1.0, SUPER(common), SUPER(alpha));
        // savematrix(&alpha, "alpha");
        // solve(common,beta) | beta => common^{-1} x beta
        dplasma_dtrsm(parsec,dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaUnit, 1.0, SUPER(common), SUPER(beta));
        dplasma_dtrsm(parsec,dplasmaLeft, dplasmaUpper, dplasmaNoTrans, dplasmaNonUnit, 1.0, SUPER(common), SUPER(beta));
        flops += FLOPS_DGETRF(N, N) + 2 * FLOPS_DTRSM(dplasmaLeft, N, N);
        // savematrix(&beta, "beta");
        // gamma   = gamma + alpha.T@gamma@solve(common,alpha)
        dplasma_dgemm(parsec,dplasmaTrans, dplasmaNoTrans, 1.0, SUPER(alphaold), SUPER(gamma), 0.0, SUPER(ws2));
        flops += FLOPS_DGEMM(N, N, N);
        // savematrix(&ws2, "desbuf1");
        double norm;
        norm = dplasma_dlange(parsec,dplasmaFrobeniusNorm, SUPER(ws2));
        parsec_context_start( parsec );
        parsec_context_wait( parsec );
        printf("iter %d norm descbuf1 %.6e \n", iter, norm);
        if (norm < 1e-15)
            break;
        dplasma_dlacpy(parsec,dplasmaUpperLower, SUPER(gamma), SUPER(gammaold));
        dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(gamma), SUPER(alpha), 0.0, SUPER(ws1));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dgemm(parsec,dplasmaTrans, dplasmaNoTrans, 1.0, SUPER(alphaold), SUPER(ws1), 0.0, SUPER(ws2));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dgeadd(parsec,dplasmaNoTrans, 1.0, SUPER(ws2), 1.0, SUPER(gamma));
        dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaTrans, 1.0, SUPER(beta), SUPER(alphaold), 0.0, SUPER(ws3));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(alphaold), SUPER(ws3), 0.0, SUPER(ws4));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dgeadd(parsec,dplasmaNoTrans, 1.0, SUPER(ws4), 1.0, SUPER(betaold));
        dplasma_dlacpy(parsec,dplasmaUpperLower, SUPER(betaold), SUPER(beta));
        dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(alphaold), SUPER(alpha), 0.0, SUPER(ws5));
        flops += FLOPS_DGEMM(N, N, N);
        dplasma_dlacpy(parsec,dplasmaUpperLower, SUPER(ws5), SUPER(alpha));
        dplasma_dgeadd(parsec,dplasmaNoTrans, -1.0, SUPER(alpha), 1.0, SUPER(alphaold));
        
    }
    dplasma_dgemm(parsec, dplasmaTrans, dplasmaNoTrans, 1.0, SUPER(Bt), SUPER(gamma), 0.0, SUPER(ws7));
    dplasma_dgemm(parsec, dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(ws7), SUPER(Bt), 1.0, SUPER(Rt));
    info = dplasma_dgetrf_nopiv( parsec,SUPER(Rt));
    dplasma_dtrsm(parsec,dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaUnit, 1.0, SUPER(Rt), SUPER(ws7));
    dplasma_dtrsm(parsec,dplasmaLeft, dplasmaUpper, dplasmaNoTrans, dplasmaNonUnit, 1.0, SUPER(Rt), SUPER(ws7));
    dplasma_dgemm(parsec,dplasmaNoTrans, dplasmaNoTrans, 1.0, SUPER(ws7), SUPER(At), 0.0, SUPER(Kout));
    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    char koutname[500];
    flops = 1e-9 * flops;
    double gflops = flops / (elapsed_time * 1e-6);
    printf("Time %f GFlops %f \n", elapsed_time * 1e-6, gflops);
    sprintf(koutname, "%s/descK%d-double-dplasma-lunopiv.fits", datapath.c_str(), N);
    
    // write_desc<double>(koutname, Kout);
    parsec_fini(&parsec);
#ifdef PARSEC_HAVE_MPI
    /** Finalize MPI */
    MPI_Finalize();
#endif
    return 0;
}

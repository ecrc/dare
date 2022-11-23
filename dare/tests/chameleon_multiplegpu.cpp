// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

/**
 * This file is used to test chameleon correctness on multiple gpus.
 * The only dependencies is chameleon.
 * */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/resource.h>
#include <coreblas/lapacke.h>
#include <chameleon.h>
#include <chameleon/timer.h>
#include <chameleon/flops.h>
#include <iostream>

#include "chameleonsrc/chameio.hpp"
#include "chameleonsrc/testings.h"

using std::cout;
using std::endl;
using std::to_string;

static void get_thread_count(int *thrdnbr)
{
    *thrdnbr = sysconf(_SC_NPROCESSORS_ONLN);
}

static int startswith(const char *s, const char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(s, prefix, n))
        return 0;
    return 1;
}

enum iparam_step1
{
    IPARAM_THRDNBR,   /* Number of cores                            */
    IPARAM_NGPUS,     /* Number of gpus                             */
    IPARAM_N,         /* #Samples x #Layers                         */
    IPARAM_NINSTR,    /* Instrument dimensions                      */
    IPARAM_NB,        /* Tile size                                  */
    IPARAM_IB,        /* Inner blocking size                        */
    IPARAM_SYNC,      /* Synchronous execution                      */
    IPARAM_NOOPTALGO, /* Algorithmic optimization                   */
    IPARAM_PROFILE,   /* Kernel profiling                           */
    IPARAM_CHECK,     /* Check correctness                          */
    /* End */
    IPARAM_SIZEOF
};

static void init_iparam(int iparam[IPARAM_SIZEOF])
{
    iparam[IPARAM_THRDNBR] = -1;
    iparam[IPARAM_NGPUS] = 0;
    iparam[IPARAM_N] = 500;
    iparam[IPARAM_NINSTR] = 500;
    iparam[IPARAM_NB] = 128;
    iparam[IPARAM_IB] = 32;
    iparam[IPARAM_SYNC] = 0;
    iparam[IPARAM_NOOPTALGO] = 0;
    iparam[IPARAM_PROFILE] = 0;
    iparam[IPARAM_CHECK] = 0;
}

static void show_help(char *prog_name)
{
    printf("Usage:\n%s [options]\n\n", prog_name);
    printf("Options are:\n"
           "  --help           Show this help\n"
           "\n"
           "  --n=X            #samples x #layers. states    (default: 500)\n"
           "  --ninstr=X       Instrument dimension. measurements (default: 500)\n"
           "  --nb=X           tile size.            (default: 128)\n"
           "\n"
           "  --datapath=X     path to the data folder"
           "                   this folder must contain the following files:"
           "                       At.fits"
           "                       BinvRt.fits"
           "                       Btinitial.fits"
           "                       Rt.fits"
           "                       Qt.fits"
           "                   the parameters '--n' and '--ninstr' will be overwritten by the"
           "                   corresponding dimensions of the files"
           "\n"
           "  --threads=X      Number of CPU workers (default: _SC_NPROCESSORS_ONLN)\n"
           "  --gpus=X         Number of GPU workers (default: 0)\n"
           "  --sync           Synchronize the execution of all calls (default: async)\n"
           "  --nooptalgo        Leverage matrix structure in the algorithm (default: optalgo)\n"
           "  --profile        Profile the execution of all calls (default: no profile)\n"
           "  --check          Check numerical correctness (default: no check)\n"
           "\n");
}

static void read_args(int argc, char *argv[], int *iparam, std::string &datapath)
{
    int i;
    for (i = 1; i < argc && argv[i]; ++i)
    {
        if (startswith(argv[i], "--help") || startswith(argv[i], "-help") ||
            startswith(argv[i], "--h") || startswith(argv[i], "-h"))
        {
            show_help(argv[0]);
            exit(0);
        }
        else if (startswith(argv[i], "--n="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_N]));
        }
        else if (startswith(argv[i], "--ninstr="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_NINSTR]));
        }
        else if (startswith(argv[i], "--nb="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_NB]));
        }
        else if (startswith(argv[i], "--ib="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_IB]));
        }
        else if (startswith(argv[i], "--threads="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_THRDNBR]));
        }
        else if (startswith(argv[i], "--gpus="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_NGPUS]));
        }
        else if (startswith(argv[i], "--datapath="))
        {
            datapath = std::string(argv[i]);
            datapath.erase(0, 11); // remove '--datapath='
        }
        else if (startswith(argv[i], "--sync"))
        {
            iparam[IPARAM_SYNC] = 1;
        }
        else if (startswith(argv[i], "--nooptalgo"))
        {
            iparam[IPARAM_NOOPTALGO] = 1;
        }
        else if (startswith(argv[i], "--profile"))
        {
            iparam[IPARAM_PROFILE] = 1;
        }
        else if (startswith(argv[i], "--check"))
        {
            iparam[IPARAM_CHECK] = 1;
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }
    }
}

static void print_header(char *prog_name, int *iparam, std::string &datapath)
{
    double eps = LAPACKE_dlamch_work('e');

    printf("#\n"
           "# DARE: %s\n"
           "# Nb threads: %d\n"
           "# Nb gpus:    %d\n"
           "# datapath    %s\n"
           "# N:          %d\n"
           "# NINSTR:     %d\n"
           "# NB:         %d\n"
           "# IB:         %d\n"
           "# Sync:       %d\n"
           "# Optalgo:    %d\n"
           "# Profile:    %d\n"
           "# Check:      %d\n"
           "# eps:        %e\n"
           "#\n",
           prog_name,
           iparam[IPARAM_THRDNBR],
           iparam[IPARAM_NGPUS],
           datapath.c_str(),
           iparam[IPARAM_N],
           iparam[IPARAM_NINSTR],
           iparam[IPARAM_NB],
           iparam[IPARAM_IB],
           iparam[IPARAM_SYNC],
           iparam[IPARAM_NOOPTALGO],
           iparam[IPARAM_PROFILE],
           iparam[IPARAM_CHECK],
           eps);

    printf("#      N  NINSTR      NB      IB    sync nooptalgo  profile   check   seconds       Gflop/s\n");
    printf("#%7d %7d %7d %7d %7d   %7d  %7d %7d\n", iparam[IPARAM_N], iparam[IPARAM_NINSTR], iparam[IPARAM_NB], iparam[IPARAM_IB], iparam[IPARAM_SYNC], iparam[IPARAM_NOOPTALGO], iparam[IPARAM_PROFILE], iparam[IPARAM_CHECK]);
    fflush(stdout);
    return;
}



#define REGVAL_DOUBLE(NAME, filename, M, N) \
    CHAM_desc_t * NAME; \
    CHAMELEON_Desc_Create(&(NAME), (void *)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, M, N, 0, 0, M, N, P, Q); \
    if(filename != "") \
    { \
        FitsFile fitsfile_ ## NAME; \
        fullpath = prefix + "/" + filename; \
        (fitsfile_ ## NAME).set(fullpath.c_str(), "r"); \
        if((fitsfile_ ## NAME).get_data_shape(0) != std::vector<long>{M,N}) {printf("error reading dim %s \n", filename); exit(1);}\
        read_desc<double>((fitsfile_ ## NAME), (NAME)); \
    }else{ \
        CHAMELEON_dlaset_Tile(ChamUpperLower, 0.0, 0.0, NAME); \
    }
    
#define REGVAL_GEMMWS(NAME, T1, T2, A, B, C) void * (ptr_ ## NAME) = CHAMELEON_dgemm_WS_Alloc( T1, T2,  A,  B,  C ); wsvector.push_back(ptr_ ## NAME);

int main(int argc, char ** argv)
{
    int it, maxit = 20;
    double norm = -1;
    double tol = 1e-15;
    int N;             // matrix order
    int NINSTR;        // number of RHS vectors
    int NB;            // tile size
    int IB;            // inner blocking size
    int NCPUS;            // number of cores to use
    int NGPUS;            // number of gpus (cuda devices) to use
    int UPLO = ChamLower; // where is stored L
    int P = 1, Q = 1;     // MPI process grid
    intptr_t mtxfmt = 1;  // Change the way the matrix is stored (0: global, 1: tiles, 2: OOC)
    int seedA = 1, seedB = 2, seedR = 3;            // seed for random number generation
    int seedalpha = 4, seedbeta = 5, seedgamma = 6; // seed for random number generation
    int trace = 0;
    int iparam[IPARAM_SIZEOF];
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int));
    init_iparam(iparam);
    std::string datapath = "";
    read_args(argc, argv, iparam, datapath);
    N = iparam[IPARAM_N];
    NINSTR = iparam[IPARAM_NINSTR];
    NB = iparam[IPARAM_NB];
    IB = iparam[IPARAM_IB];
    if (iparam[IPARAM_THRDNBR] == -1)
    {
        get_thread_count(&(iparam[IPARAM_THRDNBR]));
    }
    NCPUS = iparam[IPARAM_THRDNBR];
    NGPUS = iparam[IPARAM_NGPUS];
    int check = iparam[IPARAM_CHECK];
    int sync = iparam[IPARAM_SYNC];
    int nooptalgo = iparam[IPARAM_NOOPTALGO];
    int profile = iparam[IPARAM_PROFILE];

    /* print informations to user */
    print_header(argv[0], iparam, datapath);
    
    cout << "hello" << endl;
    // init
    CHAMELEON_Init(NCPUS, NGPUS);
    CHAMELEON_Set(CHAMELEON_TILE_SIZE, NB);
    CHAMELEON_Set(CHAMELEON_INNER_BLOCK_SIZE, IB);
    CHAMELEON_Set(CHAMELEON_HOUSEHOLDER_MODE, ChamFlatHouseholder);

    // read data
    std::string prefix, fullpath;
    std::vector<void*> wsvector;
    std::vector<CHAM_desc_t*> descvector;
    // prefix = "/home/hongy0a/research/dare-dev/build";
    // REGVAL_DOUBLE(common, "common_transpose1.fits", N, N);
    prefix = "/home/hongy0a/data/astronomydare/7090";
    
    
    REGVAL_DOUBLE(At, "At7090_transpose.fits", N, N);
    REGVAL_DOUBLE(Bt, "Btinitial7090_transpose.fits", N, NINSTR );
    REGVAL_DOUBLE(BinvRt, "BinvRt7090_transpose.fits", N, NINSTR );
    REGVAL_DOUBLE(Qt, "Qt7090.fits", N, N );
    REGVAL_DOUBLE(Rt, "Rt7090.fits", NINSTR, NINSTR );
    
    
    REGVAL_DOUBLE(alpha, "", N, N );
    REGVAL_DOUBLE(alphaold, "", N, N);
    REGVAL_DOUBLE(beta, "", N, N );
    REGVAL_DOUBLE(betaold, "", N, N );
    REGVAL_DOUBLE(Id, "", N, N );
    REGVAL_DOUBLE(gamma, "", N, N );
    REGVAL_DOUBLE(commonxalphaold, "", N, N);
    REGVAL_DOUBLE(common, "", N, N);
    REGVAL_DOUBLE(commonold, "", N, N);
    REGVAL_DOUBLE(buf1, "", N, N);
    REGVAL_DOUBLE(buf2, "", N, N);
    REGVAL_DOUBLE(buf3, "", N, N);
    REGVAL_GEMMWS(ws1, ChamNoTrans, ChamTrans, BinvRt, BinvRt, beta);
    REGVAL_GEMMWS(ws2, ChamNoTrans, ChamNoTrans, beta, gamma, common);
    REGVAL_GEMMWS(ws3, ChamTrans, ChamTrans, alphaold, gamma, buf1);
    REGVAL_GEMMWS(ws4, ChamNoTrans, ChamTrans, buf1, alpha, gamma);
    REGVAL_GEMMWS(ws5, ChamNoTrans, ChamTrans, betaold, alphaold, buf2);
    REGVAL_GEMMWS(ws6, ChamNoTrans, ChamNoTrans, alphaold, buf2, beta);
    REGVAL_GEMMWS(ws7, ChamNoTrans, ChamTrans, alphaold, alpha, buf3);
    REGVAL_GEMMWS(ws8, ChamNoTrans, ChamNoTrans, commonold, betaold, commonxalphaold);
    
    // create sequence 
    RUNTIME_sequence_t *sequence = NULL;
    RUNTIME_request_t request = RUNTIME_REQUEST_INITIALIZER;
    CHAMELEON_Sequence_Create( &sequence );
    // real compute starts here
    CHAMELEON_dlaset_Tile_Async(ChamUpperLower, 0.0, 1.0, Id, sequence, &request);
    CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamTrans, 1.0, BinvRt, BinvRt, 0.0, beta, ptr_ws1, sequence, &request);
    CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, Qt, gamma, sequence, &request );
    CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, At, alpha, sequence, &request );

    for(int it=0; it < 50; it++)
    {
        //generate common
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans, 1.0, beta, gamma, 0.0, common, ptr_ws2, sequence, &request);
        CHAMELEON_dgeadd_Tile_Async( ChamNoTrans, 1.0, Id, 1.0, common, sequence, &request );
        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, alpha, alphaold, sequence, &request );
        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, common, commonold, sequence, &request );
        CHAMELEON_dgesv_nopiv_Tile_Async( common, alpha, sequence, &request );
        CHAMELEON_dgemm_Tile_Async( ChamTrans, ChamNoTrans, 1.0, alphaold, gamma, 0.0, buf1, ptr_ws3, sequence, &request);
        CHAMELEON_dlange_Tile_Async( ChamFrobeniusNorm, buf1, &norm, sequence, &request);
        CHAMELEON_Sequence_Wait( sequence );
        printf("norm value is %.6e \n", norm);
        // update gamma
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamTrans, 1.0, buf1, alpha, 1.0, gamma, ptr_ws4, sequence, &request);
        // update beta
        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, beta, betaold, sequence, &request );
        CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamLower, ChamNoTrans, ChamUnit, (double)1.0, common, betaold, sequence, &request );
        CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamUpper, ChamNoTrans, ChamNonUnit, (double)1.0, common, betaold, sequence, &request );

        CHAMELEON_dgemm_Tile_Async(ChamNoTrans, ChamNoTrans, 1.0, commonold, betaold, 0.0, \
        commonxalphaold, ptr_ws8, sequence, &request);
        CHAMELEON_dgeadd_Tile_Async(ChamNoTrans, -1.0, beta, 1.0, commonxalphaold, sequence, &request);
        CHAMELEON_dlange_Tile_Async( ChamFrobeniusNorm, commonxalphaold, &norm, sequence, &request);
        CHAMELEON_Sequence_Wait( sequence );
        printf("norm value of betaold decomposition is %.6e \n", norm);

        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamTrans,1.0, betaold, alphaold, 0.0, buf2, ptr_ws5, sequence, &request );
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,1.0, alphaold, buf2, 1.0, beta, ptr_ws6, sequence, &request );
        // update alpha
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,1.0, alphaold, alpha, 0.0, buf3, ptr_ws7, sequence, &request );
        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, buf3, alpha, sequence, &request );
        // break;
    }

    CHAMELEON_Sequence_Destroy( sequence );
    for(int i=0; i<wsvector.size(); i++) CHAMELEON_dgemm_WS_Free(wsvector[i]);
    for(int i=0; i<descvector.size(); i++) CHAMELEON_Desc_Destroy(&descvector[i]);
    CHAMELEON_Finalize();
    return 0;
}
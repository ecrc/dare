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
#include "chameleonsrc/testings.h"
#include "chameleonsrc/chameio.hpp"

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

/* Input parameters */
enum iparam_step1
{
    IPARAM_THRDNBR,   /* Number of cores                            */
    IPARAM_NGPUS,     /* Number of gpus                             */
    IPARAM_M,         /* #Samples x #Layers                         */
    IPARAM_N,         /* #Samples x #Layers                         */
    IPARAM_K,    /* Instrument dimensions                      */
    IPARAM_NB,        /* Tile size                                  */
    IPARAM_IB,        /* Inner blocking size                        */
    IPARAM_SIZEOF
};

/**
 * Initialize integer parameters
 */
static void init_iparam(int iparam[IPARAM_SIZEOF])
{
    iparam[IPARAM_THRDNBR] = -1;
    iparam[IPARAM_NGPUS] = 0;
    iparam[IPARAM_M] = 500;
    iparam[IPARAM_N] = 500;
    iparam[IPARAM_K] = 500;
    iparam[IPARAM_NB] = 128;
    iparam[IPARAM_IB] = 32;
}

/**
 * Print how to use the program
 */
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

/**
 * Read arguments following step1 program call
 */
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
        else if (startswith(argv[i], "--M="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_M]));
        }
        else if (startswith(argv[i], "--N="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_N]));
        }
        else if (startswith(argv[i], "--K="))
        {
            sscanf(strchr(argv[i], '=') + 1, "%d", &(iparam[IPARAM_K]));
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
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
        }
    }
}


int main(int argc, char *argv[])
{
    int it, maxit = 20;
    double norm = -1;
    double tol = 1e-15;
    size_t M;             // matrix order
    size_t N;        // number of RHS vectors
    size_t K;        // number of RHS vectors
    size_t NB;            // tile size
    size_t IB;            // inner blocking size
    int NCPUS;            // number of cores to use
    int NGPUS;            // number of gpus (cuda devices) to use
    int P = 1, Q = 1;     // MPI process grid
    intptr_t mtxfmt = 1;  // Change the way the matrix is stored (0: global, 1: tiles, 2: OOC)

    int seedA = 1, seedB = 2, seedC = 3;            // seed for random number generation

    CHAM_desc_t *descA, *descB, *descC;

    /* declarations to time the program and evaluate performances */
    double flops = 0.0, gflops, time;

    /* initialize some parameters with default values */
    int iparam[IPARAM_SIZEOF];
    memset(iparam, 0, IPARAM_SIZEOF * sizeof(int));
    init_iparam(iparam);

    /* read arguments */
    std::string datapath = "";
    read_args(argc, argv, iparam, datapath);
    M = iparam[IPARAM_M];
    N = iparam[IPARAM_N];
    K = iparam[IPARAM_K];
    printf("dim M %d N %d K %d \n", M,N, K);

    NB = iparam[IPARAM_NB];
    IB = iparam[IPARAM_IB];

    /* compute the algorithm complexity to evaluate performances */
    time = 0.0;

    /* initialize the number of thread if not given by the user in argv
     * It makes sense only if this program is linked with pthread and
     * multithreaded BLAS and LAPACK */
    if (iparam[IPARAM_THRDNBR] == -1)
    {
        get_thread_count(&(iparam[IPARAM_THRDNBR]));
    }
    NCPUS = iparam[IPARAM_THRDNBR];
    NGPUS = iparam[IPARAM_NGPUS];

    /* Initialize CHAMELEON with main parameters */
    CHAMELEON_Init(NCPUS, NGPUS);
    CHAMELEON_Set(CHAMELEON_TILE_SIZE, NB);
    CHAMELEON_Set(CHAMELEON_INNER_BLOCK_SIZE, IB);
    CHAMELEON_Set(CHAMELEON_HOUSEHOLDER_MODE, ChamFlatHouseholder);

    /* CHAMELEON sequence uniquely identifies a set of asynchronous function calls
     * sharing common exception handling */
    RUNTIME_sequence_t *sequence = NULL;
    /* CHAMELEON request uniquely identifies each asynchronous function call */
    RUNTIME_request_t request = RUNTIME_REQUEST_INITIALIZER;

    CHAMELEON_Sequence_Create(&sequence);

    /* Creates the matrices */
    CHAMELEON_Desc_Create(
        &descA, (void *)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, M, K, 0, 0, M, K, P, Q);
    CHAMELEON_Desc_Create(
        &descB, (void *)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, K,N, 0, 0, K,N, P, Q);
    CHAMELEON_Desc_Create(
        &descC, (void *)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, M,N, 0, 0, M,N, P, Q);
    void *ws1;
    ws1 = CHAMELEON_dgemm_WS_Alloc(ChamNoTrans, ChamTrans, descA, descB, descC);

    /* Fills the matrix with random values */
    CHAMELEON_dplrnt_Tile(descA, seedA);
    CHAMELEON_dplrnt_Tile(descB, seedB);
    CHAMELEON_dplrnt_Tile(descC, seedC);

    START_TIMING(time);
    CHAMELEON_dgemm_Tile_Async(ChamNoTrans, ChamNoTrans,1.0, descA, descB,
                               0.0, descC, ws1, sequence, &request);
    CHAMELEON_Sequence_Wait(sequence);
    flops = flops_dgemm(M, N, K);
    STOP_TIMING(time);

    flops = 1e-9 * flops;
    gflops = flops / time;
    printf("Chameleon GEMM M %d N %d K %d time %9.3f GFLOPS %9.2f\n", M, N,K,time, gflops);
    fflush(stdout);
    CHAMELEON_Sequence_Destroy(sequence);

    CHAMELEON_dgemm_WS_Free(ws1);

    CHAMELEON_Desc_Destroy(&descA);
    CHAMELEON_Desc_Destroy(&descB);
    CHAMELEON_Desc_Destroy(&descC);

    /* Finalize CHAMELEON */
    CHAMELEON_Finalize();

    return EXIT_SUCCESS;
}
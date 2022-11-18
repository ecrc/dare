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

using std::to_string;

static void get_thread_count(int *thrdnbr) {
    *thrdnbr = sysconf(_SC_NPROCESSORS_ONLN);
}

static int startswith(const char *s, const char *prefix) {
    size_t n = strlen( prefix );
    if (strncmp( s, prefix, n ))
        return 0;
    return 1;
}


/* Input parameters */
enum iparam_step1 {
    IPARAM_THRDNBR,        /* Number of cores                            */
    IPARAM_NGPUS,          /* Number of gpus                             */
    IPARAM_N,              /* #Samples x #Layers                         */
    IPARAM_NINSTR,         /* Instrument dimensions                      */
    IPARAM_NB,             /* Tile size                                  */
    IPARAM_IB,             /* Inner blocking size                        */
    IPARAM_SYNC,           /* Synchronous execution                      */
    IPARAM_NOOPTALGO,      /* Algorithmic optimization                   */
    IPARAM_PROFILE,        /* Kernel profiling                           */
    IPARAM_CHECK,          /* Check correctness                          */
    /* End */
    IPARAM_SIZEOF
};

/**
 * Initialize integer parameters
 */
static void init_iparam(int iparam[IPARAM_SIZEOF]){
    iparam[IPARAM_THRDNBR       ] = -1;
    iparam[IPARAM_NGPUS         ] = 0;
    iparam[IPARAM_N             ] = 500;
    iparam[IPARAM_NINSTR        ] = 500;
    iparam[IPARAM_NB            ] = 128;
    iparam[IPARAM_IB            ] = 32;
    iparam[IPARAM_SYNC          ] = 0;
    iparam[IPARAM_NOOPTALGO     ] = 0;
    iparam[IPARAM_PROFILE       ] = 0;
    iparam[IPARAM_CHECK         ] = 0;
 }

/**
 * Print how to use the program
 */
static void show_help(char *prog_name) {
    printf( "Usage:\n%s [options]\n\n", prog_name );
    printf( "Options are:\n"
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
static void read_args(int argc, char *argv[], int *iparam, std::string &datapath){
    int i;
    for (i = 1; i < argc && argv[i]; ++i) {
        if ( startswith( argv[i], "--help") || startswith( argv[i], "-help") ||
             startswith( argv[i], "--h") || startswith( argv[i], "-h") ) {
            show_help( argv[0] );
            exit(0);
        } else if (startswith( argv[i], "--n=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_N]) );
        } else if (startswith( argv[i], "--ninstr=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_NINSTR]) );
        } else if (startswith( argv[i], "--nb=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_NB]) );
        } else if (startswith( argv[i], "--ib=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_IB]) );
        } else if (startswith( argv[i], "--threads=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_THRDNBR]) );
        } else if (startswith( argv[i], "--gpus=" )) {
            sscanf( strchr( argv[i], '=' ) + 1, "%d", &(iparam[IPARAM_NGPUS]) );
        } else if (startswith( argv[i], "--datapath=" )) {
            datapath = std::string(argv[i]);
            datapath.erase(0, 11); //remove '--datapath='
        } else if (startswith( argv[i], "--sync" )) {
            iparam[IPARAM_SYNC] = 1;
        } else if (startswith( argv[i], "--nooptalgo" )) {
            iparam[IPARAM_NOOPTALGO] = 1;
        } else if (startswith( argv[i], "--profile" )) {
            iparam[IPARAM_PROFILE] = 1;
        } else if (startswith( argv[i], "--check" )) {
            iparam[IPARAM_CHECK] = 1;
        } else {
            fprintf( stderr, "Unknown option: %s\n", argv[i] );
        }
    }
}

/**
 * Print a header message to summarize main parameters
 */
static void print_header(char *prog_name, int * iparam, std::string &datapath) {
    double  eps = LAPACKE_dlamch_work( 'e' );

    printf( "#\n"
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
            eps );

    printf( "#      N  NINSTR      NB      IB    sync nooptalgo  profile   check   seconds       Gflop/s\n");
    printf( "#%7d %7d %7d %7d %7d   %7d  %7d %7d", iparam[IPARAM_N], iparam[IPARAM_NINSTR], iparam[IPARAM_NB], iparam[IPARAM_IB], iparam[IPARAM_SYNC], iparam[IPARAM_NOOPTALGO], iparam[IPARAM_PROFILE], iparam[IPARAM_CHECK]);
    fflush( stdout );
    return;
}

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
    FitsFile &file_Btinitial, FitsFile &file_Qt, FitsFile &file_Rt, int &N, int &NINSTR){
        std::vector<long> shape_BinvRt = file_BinvRt.get_data_shape(0);
        N = shape_BinvRt[0];
        NINSTR = shape_BinvRt[1];

        // checking dimensions compatibility
        int error = 0;
        if(file_At.get_data_shape(0) != std::vector<long>{N,N}){
            error=1;
            std::cerr<<"Dimensions of At are not good"<<std::endl;
        }
        if(shape_BinvRt != std::vector<long>{N,NINSTR}){
            error=1;
            std::cerr<<"Dimensions of BinvRt are not good"<<std::endl;
        }
        if( file_Btinitial.get_data_shape(0) != std::vector<long>{N,NINSTR}){
            error=1;
            std::cerr<<"Dimensions of Binitial are not good"<<std::endl;
        }
        if( file_Qt.get_data_shape(0) != std::vector<long>{N,N}){
            error=1;
            std::cerr<<"Dimensions of Qt are not good"<<std::endl;
        }
        if( file_Rt.get_data_shape(0) != std::vector<long>{NINSTR, NINSTR}){
            error=1;
            std::cerr<<"Dimensions of Rt are not good"<<std::endl;
        }
        if(error){
            throw std::runtime_error("Matrices dimensions inconsistent");
        }
}

/*
 * test external application link with chameleon
 */
int main(int argc, char *argv[]) {

    size_t i, j;
    int it, maxit = 20;
    double norm = -1;
    double tol=1e-15;
    size_t N;    // matrix order
    size_t NINSTR; // number of RHS vectors
    size_t NB;   // tile size
    size_t IB;   // inner blocking size
    int NCPUS; // number of cores to use
    int NGPUS; // number of gpus (cuda devices) to use
    int UPLO = ChamLower; // where is stored L
    int P = 1, Q = 1; // MPI process grid
    intptr_t mtxfmt = 1; //Change the way the matrix is stored (0: global, 1: tiles, 2: OOC)

    int seedA = 1, seedB = 2, seedR = 3; // seed for random number generation
    int seedalpha = 4, seedbeta = 5, seedgamma = 6; // seed for random number generation

    int trace = 0;

    FitsFile file_At;
    FitsFile file_BinvRt;
    FitsFile file_Btinitial;
    FitsFile file_Qt;
    FitsFile file_Rt;

    CHAM_desc_t *descA, *descB, *descBinit, *descQ, *descR, *descK, *descT1, *descT2;
    CHAM_desc_t *descalpha, *descbeta, *descgamma, *desccommon, *descId;
    CHAM_desc_t *descalphaold, *descbetaold;
    CHAM_desc_t *descbuf1, *descbuf2, *descbuf3, *descbuf4, *descbuf5;

    /* declarations to time the program and evaluate performances */
    double flops = 0.0, gflops, time;

    /* variable to check the numerical results */
    double anorm, bnorm, xnorm, eps, res;
    int hres = 0; // return code for chameleon functions

    /* initialize some parameters with default values */
    int iparam[IPARAM_SIZEOF];
    memset(iparam, 0, IPARAM_SIZEOF*sizeof(int));
    init_iparam(iparam);

    /* read arguments */
    std::string datapath="";
    read_args(argc, argv, iparam, datapath);
    N      = iparam[IPARAM_N];
    NINSTR = iparam[IPARAM_NINSTR];
    NB     = iparam[IPARAM_NB];
    IB     = iparam[IPARAM_IB];
    if(datapath!= ""){
        file_At.set(datapath+"/At"+to_string(N)+"_transpose.fits","r");
        file_BinvRt.set(datapath+"/BinvRt"+to_string(N)+"_transpose.fits","r");
        file_Btinitial.set(datapath+"/Btinitial"+to_string(N)+"_transpose.fits","r");
        file_Qt.set(datapath+"/Qt"+to_string(N)+".fits","r");
        file_Rt.set(datapath+"/Rt"+to_string(N)+".fits","r");
        get_matrices_dimensions(datapath,file_At,file_BinvRt,file_Btinitial,file_Qt,file_Rt,
            iparam[IPARAM_N], iparam[IPARAM_NINSTR]);
    }


    /* compute the algorithm complexity to evaluate performances */
    time = 0.0;

    /* initialize the number of thread if not given by the user in argv
     * It makes sense only if this program is linked with pthread and
     * multithreaded BLAS and LAPACK */
    if ( iparam[IPARAM_THRDNBR] == -1 ) {
        get_thread_count( &(iparam[IPARAM_THRDNBR]) );
    }
    NCPUS = iparam[IPARAM_THRDNBR];
    NGPUS = iparam[IPARAM_NGPUS];
    int check = iparam[IPARAM_CHECK];
    int sync = iparam[IPARAM_SYNC];
    int nooptalgo = iparam[IPARAM_NOOPTALGO];
    int profile = iparam[IPARAM_PROFILE];

    /* print informations to user */
    print_header( argv[0], iparam, datapath);

    /* Initialize CHAMELEON with main parameters */
    CHAMELEON_Init( NCPUS, NGPUS );
    CHAMELEON_Set( CHAMELEON_TILE_SIZE, NB );
    CHAMELEON_Set( CHAMELEON_INNER_BLOCK_SIZE, IB );
    CHAMELEON_Set( CHAMELEON_HOUSEHOLDER_MODE, ChamFlatHouseholder );
    //RUNTIME_dlocality_allrestrict( RUNTIME_CUDA );

    //CHAM_context_t *chamctxt;
    //chamctxt = chameleon_context_self();

    /* CHAMELEON sequence uniquely identifies a set of asynchronous function calls
     * sharing common exception handling */
    RUNTIME_sequence_t *sequence = NULL;
    /* CHAMELEON request uniquely identifies each asynchronous function call */
    RUNTIME_request_t request = RUNTIME_REQUEST_INITIALIZER;

    CHAMELEON_Sequence_Create(&sequence);


    /* Creates the matrices */
    CHAMELEON_Desc_Create(
        &descA, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descB, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, NINSTR, 0, 0, N, NINSTR, P, Q );
    CHAMELEON_Desc_Create(
        &descBinit, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, NINSTR, 0, 0, N, NINSTR, P, Q );
    CHAMELEON_Desc_Create(
        &descQ, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descR, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, NINSTR, NINSTR, 0, 0, NINSTR, NINSTR, P, Q ); // descR needs to be inverted then square rooted
    CHAMELEON_Desc_Create(
        &descK, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, NINSTR, N, 0, 0, NINSTR, N, P, Q );
    CHAMELEON_Desc_Create(
        &descalpha, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descbeta, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descgamma, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descId, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &desccommon, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descalphaold, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descbetaold, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descbuf1, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );
    CHAMELEON_Desc_Create(
        &descbuf2, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, NINSTR, N, 0, 0, NINSTR, N, P, Q );
    CHAMELEON_Desc_Create(
        &descbuf3, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, NINSTR, 0, 0, N, NINSTR, P, Q );
    CHAMELEON_Desc_Create(
        &descbuf4, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, NINSTR, NINSTR, 0, 0, NINSTR, NINSTR, P, Q );
    CHAMELEON_Desc_Create(
        &descbuf5, (void*)(-mtxfmt), ChamRealDouble, NB, NB, NB * NB, N, N, 0, 0, N, N, P, Q );

    CHAMELEON_Alloc_Workspace_dgels( N, N, &descT1, P, Q );
    CHAMELEON_Alloc_Workspace_dgels( NINSTR, NINSTR, &descT2, P, Q );


    void *ws1, *ws2, *ws3, *ws4, *ws5, *ws6, *ws7, *ws8, *ws9, *ws10, *ws11, *ws12, *ws13, *ws14, *ws15;
    ws1 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamTrans, descB, descB, descbetaold );
    ws2 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbeta, descgamma, desccommon );
    ws3 = CHAMELEON_dgemm_WS_Alloc( ChamTrans, ChamNoTrans, descalphaold, descgamma, descbuf1 );
    ws4 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf1, descalpha, descgamma );
    ws5 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamTrans, descbeta, descalphaold, descbuf1 );
    ws6 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descalphaold, descbuf1, descbetaold );
    ws7 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descalphaold, descalpha, descbuf1 );
    ws8 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descBinit, descgamma, descbuf2 );
    ws9 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf2, descBinit, descR );
    ws10 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf2, descA, descK );
    ws11 = CHAMELEON_dgemm_WS_Alloc( ChamTrans, ChamNoTrans, descA, descgamma, descbuf5 );
    ws12 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf2, descBinit, descbuf4 );
    ws13 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf5, descBinit, descbuf3 );
    ws14 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf3, descK, descQ );
    ws15 = CHAMELEON_dgemm_WS_Alloc( ChamNoTrans, ChamNoTrans, descbuf5, descA, descQ );


    if(datapath != ""){
    /* Fills the matrix with files content */
        read_desc<double>(file_At, descA);
        read_desc<double>(file_BinvRt, descB);
        read_desc<double>(file_Btinitial, descBinit);
        read_desc<double>(file_Qt, descQ);
        read_desc<double>(file_Rt, descR);
    }else{
    /* Fills the matrix with random values */
        CHAMELEON_dplrnt_Tile( descA, seedA );
        CHAMELEON_dplrnt_Tile( descB, seedB );
        CHAMELEON_dplrnt_Tile( descBinit, seedB );
        CHAMELEON_dplrnt_Tile( descQ, seedR );
        CHAMELEON_dplrnt_Tile( descR, seedR );
    }


    /* Start kernel statistics */
    if ( profile ) {
        CHAMELEON_Enable( CHAMELEON_KERNELPROFILE_MODE );
    }

    /* Start tracing */
    if ( trace ) {
        CHAMELEON_Enable( CHAMELEON_PROFILING_MODE );
    }


    /* Calculates the DARE solution */
    START_TIMING( time );
    printf("\n");
    CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamTrans,
                      1.0, descB, descB,
                      0.0, descbetaold,
		      ws1,
		      sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );
    flops = flops + flops_dgemm( N, N, NINSTR );
    CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descA, descalpha, sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );
    CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descQ, descgamma, sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );
    CHAMELEON_dlaset_Tile_Async( ChamUpperLower, 0.0, 1.0, descId, sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );

    /* Create the task flag */
    //RUNTIME_option_t options;

    //RUNTIME_barrier( chamctxt );
    //RUNTIME_enable( chamctxt, CHAMELEON_DAG );
    
    for (it = 1; it <= maxit; it++) {

        //RUNTIME_barrier( chamctxt );
        //RUNTIME_options_init_color(&options, chamctxt);

        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descbetaold, descbeta, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                          1.0, descbeta, descgamma,
                          0.0, desccommon, ws2, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );
        CHAMELEON_dgeadd_Tile_Async( ChamNoTrans, 1.0, descId, 1.0, desccommon, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descalpha, descalphaold, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );

        //RUNTIME_options_init_color(&options, chamctxt);
        /*
        if (nooptalgo) {
            CHAMELEON_dgels_Tile_Async( ChamNoTrans, desccommon, descT1, descalpha, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
            flops = flops + flops_dgeqrf( N, N ) + flops_dormqr( ChamLeft, N, N, N ) + flops_dtrsm( ChamLeft, N, N );
        }
        else {
            if ( it < 7) {
                 CHAMELEON_dgesv_nopiv_Tile_Async( desccommon, descalpha, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dgetrf( N, N ) + flops_dgetrs( N, N );
            }
            else {
                 CHAMELEON_dgels_Tile_Async( ChamNoTrans, desccommon, descT1, descalpha, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dgeqrf( N, N ) + flops_dormqr( ChamLeft, N, N, N ) + flops_dtrsm( ChamLeft, N, N );
            }
        }
        */
                 CHAMELEON_dgesv_nopiv_Tile_Async( desccommon, descalpha, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dgetrf( N, N ) + flops_dgetrs( N, N );

        //RUNTIME_options_init_color(&options, chamctxt);
        CHAMELEON_dgemm_Tile_Async( ChamTrans, ChamNoTrans,
                          1.0, descalphaold, descgamma,
                          0.0, descbuf1, ws3, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );

        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                          1.0, descbuf1, descalpha,
			  1.0, descgamma, ws4, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );

	if (check) {
            //err = norm(gamma-(A'*gamma*A - (A'*gamma*B)*inv(R+B'*gamma*B)*(B'*gamma*A) + Q))
            CHAMELEON_dgemm_Tile_Async( ChamTrans, ChamNoTrans,
                   1.0, descBinit, descgamma,
                   0.0, descbuf2, ws8, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
            CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descR, descbuf4, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf2 );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err1 %e ", norm);
            CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                              1.0, descbuf2, descBinit,
                              1.0, descbuf4, ws12, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf4 );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err2 %e ", norm);
            CHAMELEON_dgels_Tile_Async( ChamNoTrans, descbuf4,
                              descT2, descbuf2, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf2 );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err3 %e ", norm);
            CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                              1.0, descbuf2, descA,
                              0.0, descK, ws10, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descK );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err4 %e ", norm);
            CHAMELEON_dgemm_Tile_Async( ChamTrans, ChamNoTrans,
                   1.0, descA, descgamma,
                   0.0, descbuf5, ws11, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf5 );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err5 %e ", norm);
            CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                   1.0, descbuf5, descBinit,
                   0.0, descbuf3, ws13, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf3 );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err6 %e ", norm);
            read_desc<double>(datapath + "/Qt.fits", descQ);
            CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                  -1.0, descbuf3, descK,
                   1.0, descQ, ws14, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descQ );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err7 %e ", norm);
            CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                   1.0, descbuf5, descA,
                   1.0, descQ, ws15, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descQ );
            //if (sync) CHAMELEON_Sequence_Wait( sequence );
	    //printf("err8 %e ", norm);
            CHAMELEON_dgeadd_Tile_Async( ChamNoTrans, 1.0, descgamma, -1.0, descQ, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descQ );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
	    printf("||LHS-RHS||f = %e ", norm);
        }

	//CHAMELEON_dlange_Tile_Async( ChamFrobeniusNorm, descbuf1, &norm, sequence, &request );
	norm = CHAMELEON_dlange_Tile( ChamFrobeniusNorm, descbuf1 );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        //CHAMELEON_Desc_Flush( descbuf1, sequence );

	printf("it %d norm %e tol %e\n", it, norm, tol);
	if (norm < tol) {
	   break;
	}

        //RUNTIME_options_init_color(&options, chamctxt);
        /*
        if (nooptalgo) {
            CHAMELEON_dormqr_Tile_Async( ChamLeft, ChamTrans,
                              desccommon, descT1, descbeta, sequence, &request );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
            flops = flops + flops_dormqr( ChamLeft, N, N, N );
            CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamUpper, ChamNoTrans, ChamNonUnit,
                              1.0, desccommon, descbeta, sequence, &request );
            flops = flops + flops_dtrsm( ChamLeft, N, N );
            if (sync) CHAMELEON_Sequence_Wait( sequence );
        }
        else {
            if ( it < 7) {
                 CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamLower, ChamNoTrans, ChamUnit, (double)1.0, desccommon, descbeta, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamUpper, ChamNoTrans, ChamNonUnit, (double)1.0, desccommon, descbeta, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dgetrs( N, N );
            }
            else {
                 CHAMELEON_dormqr_Tile_Async( ChamLeft, ChamTrans,
                                   desccommon, descT1, descbeta, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dormqr( ChamLeft, N, N, N );
                 CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamUpper, ChamNoTrans, ChamNonUnit,
                                   1.0, desccommon, descbeta, sequence, &request );
                 flops = flops + flops_dtrsm( ChamLeft, N, N );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
            }
        }
        */
                 CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamLower, ChamNoTrans, ChamUnit, (double)1.0, desccommon, descbeta, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 CHAMELEON_dtrsm_Tile_Async( ChamLeft, ChamUpper, ChamNoTrans, ChamNonUnit, (double)1.0, desccommon, descbeta, sequence, &request );
                 if (sync) CHAMELEON_Sequence_Wait( sequence );
                 flops = flops + flops_dgetrs( N, N );

        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamTrans,
                          1.0, descbeta, descalphaold,
                          0.0, descbuf1, ws5, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );

        //RUNTIME_options_init_color(&options, chamctxt);
        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                          1.0, descalphaold, descbuf1,
                          1.0, descbetaold, ws6, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );

        CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                          1.0, descalphaold, descalpha,
                          0.0, descbuf1, ws7, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgemm( N, N, N );

        CHAMELEON_dlacpy_Tile_Async( ChamUpperLower, descbuf1, descalpha, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );

        //write_desc<double>(std::string("descalpha.fits"),descalpha);

#if 0
#endif

    }
    //RUNTIME_barrier( chamctxt );
    //RUNTIME_disable( chamctxt, CHAMELEON_DAG );

#if 0
#endif
    CHAMELEON_dgemm_Tile_Async( ChamTrans, ChamNoTrans,
                      1.0, descBinit, descgamma,
                      0.0, descbuf2, ws8, sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );
    flops = flops + flops_dgemm( NINSTR, N, N );
    CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                      1.0, descbuf2, descBinit,
                      1.0, descR, ws9, sequence, &request );
    if (sync) CHAMELEON_Sequence_Wait( sequence );
    flops = flops + flops_dgemm( NINSTR, NINSTR, N );

    /*
    CHAMELEON_dgesv_nopiv_Tile_Async( descR, descbuf2, sequence, &request );
    */

    if (nooptalgo) {
        CHAMELEON_dgels_Tile_Async( ChamNoTrans, descR,
                          descT2, descbuf2, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dgeqrf( NINSTR, NINSTR ) + flops_dormqr( ChamLeft, NINSTR, N, NINSTR ), + flops_dtrsm( ChamLeft, NINSTR, N );
    }
    else {
        CHAMELEON_dposv_Tile_Async( ChamLower, descR, descbuf2, sequence, &request );
        if (sync) CHAMELEON_Sequence_Wait( sequence );
        flops = flops + flops_dpotrf( NINSTR ) + flops_dpotrs( NINSTR, N );
    }

    CHAMELEON_dgemm_Tile_Async( ChamNoTrans, ChamNoTrans,
                      1.0, descbuf2, descA,
                      0.0, descK, ws10, sequence, &request );
    CHAMELEON_Sequence_Wait( sequence );
    flops = flops + flops_dgemm( NINSTR, N, N );

    STOP_TIMING( time );

//    write_desc<double>(std::string(datapath + "/descK"+to_string(N)+"-double-lunopiv.fits"),descK);

    /* Stop tracing */
    if ( trace ) {
        CHAMELEON_Disable( CHAMELEON_PROFILING_MODE );
    }

    /* Stop kernel statistics and display results */
    if ( profile ) {
        CHAMELEON_Disable( CHAMELEON_KERNELPROFILE_MODE );
        RUNTIME_kernelprofile_display();
    }

    flops = 1e-9 * flops;
/*
    flops = 1e-9 * ( flops_dgemm( N, N, NINSTR ) +
                     (it-1) * (
                              flops_dgemm( N, N, N ) +
                              flops_dgeqrf( N, N ) +
                              flops_dormqr( ChamLeft, N, N, N ) +
                              flops_dtrsm( ChamLeft, N, N ) +
                              flops_dgemm( N, N, N ) +
                              flops_dgemm( N, N, N ) +
                              flops_dormqr( ChamLeft, N, N, N ) +
                              flops_dtrsm( ChamLeft, N, N ) +
                              flops_dgemm( N, N, N ) +
                              flops_dgemm( N, N, N ) +
                              flops_dgemm( N, N, N )
                              ) +
                     flops_dgemm( NINSTR, N, N ) +
                     flops_dgemm( NINSTR, NINSTR, N ) +
                     //flops_dgeqrf( NINSTR, NINSTR ) +
                     flops_dgetrf( NINSTR, NINSTR ) +
                     //flops_dormqr( ChamLeft, NINSTR, N, NINSTR ) +
                     //flops_dtrsm( ChamLeft, NINSTR, N ) +
                     flops_dgetrs( NINSTR, N ) +
                     flops_dgemm( NINSTR, N, N )
                   );
*/
    gflops = flops / time;
    printf( "Time: %9.3f GFlops: %9.2f\n", time, ( hres == CHAMELEON_SUCCESS ) ? gflops : -1);
    fflush( stdout );

    //CHAMELEON_Desc_Flush( descA, sequence );

    CHAMELEON_Sequence_Destroy( sequence );

    CHAMELEON_dgemm_WS_Free( ws1 );
    CHAMELEON_dgemm_WS_Free( ws2 );
    CHAMELEON_dgemm_WS_Free( ws3 );
    CHAMELEON_dgemm_WS_Free( ws4 );
    CHAMELEON_dgemm_WS_Free( ws5 );
    CHAMELEON_dgemm_WS_Free( ws6 );
    CHAMELEON_dgemm_WS_Free( ws7 );
    CHAMELEON_dgemm_WS_Free( ws8 );
    CHAMELEON_dgemm_WS_Free( ws9 );
    CHAMELEON_dgemm_WS_Free( ws10 );
    CHAMELEON_dgemm_WS_Free( ws11 );
    CHAMELEON_dgemm_WS_Free( ws12 );
    CHAMELEON_dgemm_WS_Free( ws13 );
    CHAMELEON_dgemm_WS_Free( ws14 );
    CHAMELEON_dgemm_WS_Free( ws15 );

    CHAMELEON_Desc_Destroy( &descA );
    CHAMELEON_Desc_Destroy( &descB );
    CHAMELEON_Desc_Destroy( &descBinit );
    CHAMELEON_Desc_Destroy( &descQ );
    CHAMELEON_Desc_Destroy( &descR );
    CHAMELEON_Desc_Destroy( &descK );
    CHAMELEON_Desc_Destroy( &descalpha );
    CHAMELEON_Desc_Destroy( &descalphaold );
    CHAMELEON_Desc_Destroy( &descbeta );
    CHAMELEON_Desc_Destroy( &descbetaold );
    CHAMELEON_Desc_Destroy( &descgamma );
    CHAMELEON_Desc_Destroy( &desccommon );
    CHAMELEON_Desc_Destroy( &descId );
    CHAMELEON_Desc_Destroy( &descT1 );
    CHAMELEON_Desc_Destroy( &descT2 );
    CHAMELEON_Desc_Destroy( &descbuf1 );
    CHAMELEON_Desc_Destroy( &descbuf2 );
    CHAMELEON_Desc_Destroy( &descbuf3 );
    CHAMELEON_Desc_Destroy( &descbuf4 );
    CHAMELEON_Desc_Destroy( &descbuf5 );


    /* Finalize CHAMELEON */
    CHAMELEON_Finalize();

    return EXIT_SUCCESS;
}
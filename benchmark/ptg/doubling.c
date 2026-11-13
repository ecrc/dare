#include <cblas.h>
#include <dplasma.h>
#include <lapacke.h>
#include <math.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "add_identity.h"

#if defined(DPLASMA_HAVE_CUDA)
#include <parsec/parsec_internal.h>
#include <parsec/mca/device/device.h>

/* DPLASMA's PTG CUDA bodies use these per-stream handles. The pinned
 * DPLASMA release exports these callbacks but does not install their header. */
void *dplasma_create_cuda_handles(void *stream, void *user);
void dplasma_destroy_cuda_handles(void *handles, void *user);
#endif

enum { A, G, H, S, X, Y, M, T, MATRIX_COUNT };

struct operation {
    parsec_taskpool_t *pool;
    void (*destroy)(parsec_taskpool_t *);
};

struct round_times {
    double factor;
    double update_h;
    double update_ag;
};

static double seconds(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + 1e-9 * time.tv_nsec;
}

static void destroy_add_identity(parsec_taskpool_t *pool)
{
    parsec_add_identity_taskpool_t *task = (parsec_add_identity_taskpool_t *)pool;
    PARSEC_OBJ_DESTRUCT(&task->arenas_datatypes[PARSEC_add_identity_DEFAULT_ADT_IDX]);
    parsec_taskpool_free(pool);
}

static parsec_taskpool_t *add_identity(parsec_tiled_matrix_t *s)
{
    parsec_add_identity_taskpool_t *task = parsec_add_identity_new(s);
    if (task != NULL) {
        parsec_arena_datatype_set_type(
            &task->arenas_datatypes[PARSEC_add_identity_DEFAULT_ADT_IDX],
            (size_t)s->bsiz * sizeof(double), PARSEC_ARENA_ALIGNMENT_SSE,
            PARSEC_DATATYPE_NULL);
    }
    return (parsec_taskpool_t *)task;
}

static void run(parsec_context_t *context, const struct operation *operations, size_t count)
{
    parsec_taskpool_t *chain = NULL;
    for (size_t i = 0; i < count; ++i) {
        if (operations[i].pool == NULL) {
            fputs("Could not create a PTG taskpool\n", stderr);
            exit(EXIT_FAILURE);
        }
        chain = parsec_compose(chain, operations[i].pool);
    }
    if (parsec_context_add_taskpool(context, chain) != PARSEC_SUCCESS ||
        parsec_context_start(context) != PARSEC_SUCCESS ||
        parsec_context_wait(context) != PARSEC_SUCCESS) {
        fputs("PTG execution failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    if (count > 1)
        parsec_taskpool_free(chain);
    for (size_t i = 0; i < count; ++i)
        operations[i].destroy(operations[i].pool);
}

static struct operation gemm(dplasma_enum_t left, dplasma_enum_t right,
                             parsec_tiled_matrix_t *a, parsec_tiled_matrix_t *b,
                             double beta, parsec_tiled_matrix_t *c)
{
    return (struct operation){
        dplasma_dgemm_New(left, right, 1.0, a, b, beta, c),
        dplasma_dgemm_Destruct
    };
}

static struct operation trsm(dplasma_enum_t uplo, dplasma_enum_t diag,
                             parsec_tiled_matrix_t *lu, parsec_tiled_matrix_t *rhs)
{
    return (struct operation){
        dplasma_dtrsm_New(dplasmaLeft, uplo, dplasmaNoTrans, diag, 1.0, lu, rhs),
        dplasma_dtrsm_Destruct
    };
}

static int doubling(parsec_context_t *context, parsec_tiled_matrix_t **d,
                    double *norm, struct round_times *times)
{
    int info = 0;
    const dplasma_enum_t nt = dplasmaNoTrans;
    const dplasma_enum_t tr = dplasmaTrans;
    double start = seconds();
    const struct operation factor[] = {
        gemm(nt, nt, d[G], d[H], 0.0, d[S]),
        {add_identity(d[S]), destroy_add_identity},
        {dplasma_dgetrf_nopiv_New(d[S], &info), dplasma_dgetrf_nopiv_Destruct},
    };
    run(context, factor, sizeof(factor) / sizeof(factor[0]));
    times->factor = seconds() - start;
    if (info != 0)
        return info;

    start = seconds();
    const struct operation update_h[] = {
        {dplasma_dlacpy_New(dplasmaUpperLower, d[A], d[X]), dplasma_dlacpy_Destruct},
        trsm(dplasmaLower, dplasmaUnit, d[S], d[X]),
        trsm(dplasmaUpper, dplasmaNonUnit, d[S], d[X]),
        gemm(tr, nt, d[A], d[H], 0.0, d[M]),
        gemm(nt, nt, d[M], d[X], 1.0, d[H]),
        {dplasma_dlange_New(dplasmaFrobeniusNorm, d[M], norm), dplasma_dlange_Destruct},
    };
    run(context, update_h, sizeof(update_h) / sizeof(update_h[0]));
    times->update_h = seconds() - start;
    if (*norm < 1e-15)
        return 0;

    start = seconds();
    const struct operation update_ag[] = {
        {dplasma_dlacpy_New(dplasmaUpperLower, d[G], d[Y]), dplasma_dlacpy_Destruct},
        trsm(dplasmaLower, dplasmaUnit, d[S], d[Y]),
        trsm(dplasmaUpper, dplasmaNonUnit, d[S], d[Y]),
        gemm(nt, tr, d[Y], d[A], 0.0, d[T]),
        gemm(nt, nt, d[A], d[T], 1.0, d[G]),
        gemm(nt, nt, d[A], d[X], 0.0, d[T]),
    };
    run(context, update_ag, sizeof(update_ag) / sizeof(update_ag[0]));
    times->update_ag = seconds() - start;

    parsec_tiled_matrix_t *previous_a = d[A];
    d[A] = d[T];
    d[T] = previous_a;
    return 0;
}

static double initial_value(int which, int i, int j, int n)
{
    if (which == A)
        return (i == j ? 0.97 : 0.0) + 0.04 * sin(3.0 * i + 7.0 * j + 1.0) / n;
    const double scale = which == G ? 0.22 : 0.35;
    return scale * ((i == j ? 1.0 : 0.0) + cos(2.0 * (i - j)) / n);
}

static int reference(int n, double **d, double *norm, lapack_int *pivots)
{
    const size_t bytes = (size_t)n * n * sizeof(double);
    const CBLAS_TRANSPOSE nt = CblasNoTrans;
    const CBLAS_TRANSPOSE tr = CblasTrans;

    cblas_dgemm(CblasColMajor, nt, nt, n, n, n, 1.0, d[G], n, d[H], n, 0.0, d[S], n);
    for (int i = 0; i < n; ++i)
        d[S][i + (size_t)i * n] += 1.0;
    memcpy(d[X], d[A], bytes);
    memcpy(d[Y], d[G], bytes);
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, d[S], n, pivots);
    if (info == 0)
        info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, n, d[S], n, pivots, d[X], n);
    if (info != 0)
        return info;
    cblas_dgemm(CblasColMajor, tr, nt, n, n, n, 1.0, d[A], n, d[H], n, 0.0, d[M], n);
    *norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, d[M], n);
    cblas_dgemm(CblasColMajor, nt, nt, n, n, n, 1.0, d[M], n, d[X], n, 1.0, d[H], n);
    if (*norm < 1e-15)
        return 0;
    info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, n, d[S], n, pivots, d[Y], n);
    if (info != 0)
        return info;
    cblas_dgemm(CblasColMajor, nt, tr, n, n, n, 1.0, d[Y], n, d[A], n, 0.0, d[T], n);
    cblas_dgemm(CblasColMajor, nt, nt, n, n, n, 1.0, d[A], n, d[T], n, 1.0, d[G], n);
    cblas_dgemm(CblasColMajor, nt, nt, n, n, n, 1.0, d[A], n, d[X], n, 0.0, d[T], n);
    double *previous_a = d[A];
    d[A] = d[T];
    d[T] = previous_a;
    return 0;
}

static double *tile_at(parsec_tiled_matrix_t *matrix, int row, int column)
{
    parsec_data_collection_t *dc = &matrix->super;
    parsec_data_t *data = dc->data_of(dc, row, column);
    return parsec_data_get_ptr(data, 0);
}

static double relative_error(parsec_tiled_matrix_t *matrix, const double *ref)
{
    double diff = 0.0;
    double magnitude = 0.0;
    for (int col = 0; col < matrix->nt; ++col) {
        const int ncols = col == matrix->nt - 1 ? matrix->n - col * matrix->nb : matrix->nb;
        for (int row = 0; row < matrix->mt; ++row) {
            const int nrows = row == matrix->mt - 1 ? matrix->m - row * matrix->mb : matrix->mb;
            const double *tile = tile_at(matrix, row, col);
            for (int j = 0; j < ncols; ++j) {
                for (int i = 0; i < nrows; ++i) {
                    const double expected = ref[row * matrix->mb + i +
                        (size_t)(col * matrix->nb + j) * matrix->m];
                    const double delta = tile[i + (size_t)j * matrix->mb] - expected;
                    diff += delta * delta;
                    magnitude += expected * expected;
                }
            }
        }
    }
    return sqrt(diff) / fmax(sqrt(magnitude), 1e-300);
}

static int positive_int(const char *argument)
{
    char *end = NULL;
    long value = strtol(argument, &end, 10);
    return end != argument && *end == '\0' && value > 0 && value <= 65536 ? (int)value : 0;
}

int main(int argc, char **argv)
{
    int n = 97, nb = 24, iterations = 12, cores = 4, verify = 1;
    int parsec_arg = argc;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--") == 0) {
            parsec_arg = i;
            break;
        }
        if (strcmp(argv[i], "--no-verify") == 0) {
            verify = 0;
            continue;
        }
        int *option = NULL;
        if (strcmp(argv[i], "-n") == 0) option = &n;
        if (strcmp(argv[i], "-b") == 0) option = &nb;
        if (strcmp(argv[i], "-i") == 0) option = &iterations;
        if (strcmp(argv[i], "-c") == 0) option = &cores;
        if (option == NULL || i + 1 >= argc || (*option = positive_int(argv[++i])) == 0) {
            fprintf(stderr, "Usage: %s [-n size] [-b tile] [-i rounds] [-c cores] "
                            "[--no-verify] [-- PaRSEC options]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (verify && n > 1024) {
        fputs("Reference checking is limited to n <= 1024. Use --no-verify to time larger inputs.\n", stderr);
        return EXIT_FAILURE;
    }

#if defined(PARSEC_HAVE_MPI)
    int provided, ranks;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 1 || provided < MPI_THREAD_SERIALIZED) {
        fputs("ddareptg requires one MPI rank and MPI_THREAD_SERIALIZED.\n", stderr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
#endif
    int runtime_argc = parsec_arg < argc ? argc - parsec_arg - 1 : 0;
    char **runtime_argv = runtime_argc > 0 ? argv + parsec_arg + 1 : NULL;
    parsec_context_t *context = parsec_init(cores,
        runtime_argc > 0 ? &runtime_argc : NULL,
        runtime_argc > 0 ? &runtime_argv : NULL);
    if (context == NULL) {
        fputs("Could not initialize PaRSEC\n", stderr);
#if defined(PARSEC_HAVE_MPI)
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

    parsec_matrix_block_cyclic_t matrices[MATRIX_COUNT] = {0};
    parsec_tiled_matrix_t *d[MATRIX_COUNT] = {0};
    double *ref[MATRIX_COUNT] = {0};
    lapack_int *pivots = NULL;
    int status = EXIT_FAILURE;
    int descriptors = 0;

#if defined(DPLASMA_HAVE_CUDA)
    parsec_info_id_t cuda_info = PARSEC_INFO_ID_UNDEFINED;
    for (int dev = 0; dev < (int)parsec_nb_devices; ++dev) {
        if (parsec_mca_device_get(dev)->type == PARSEC_DEV_CUDA) {
            cuda_info = parsec_info_register(&parsec_per_stream_infos, "DPLASMA::CUDA::HANDLES",
                dplasma_destroy_cuda_handles, NULL, dplasma_create_cuda_handles, NULL, NULL);
            if (cuda_info == PARSEC_INFO_ID_UNDEFINED) {
                fputs("Could not register DPLASMA's CUDA stream handles\n", stderr);
                goto cleanup;
            }
            break;
        }
    }
#endif

    if (verify) {
        pivots = malloc((size_t)n * sizeof(*pivots));
        if (pivots == NULL)
            goto cleanup;
    }
    for (int m = 0; m < MATRIX_COUNT; ++m) {
        parsec_matrix_block_cyclic_init(&matrices[m], PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
                                       0, nb, nb, n, n, 0, 0, n, n, 1, 1, 1, 1, 0, 0);
        ++descriptors;
        d[m] = &matrices[m].super;
        size_t entries = (size_t)d[m]->nb_local_tiles * d[m]->bsiz;
        matrices[m].mat = parsec_data_allocate(entries * sizeof(double));
        if (verify)
            ref[m] = calloc((size_t)n * n, sizeof(double));
        if (matrices[m].mat == NULL || (verify && ref[m] == NULL)) {
            fputs("Could not allocate matrix buffers\n", stderr);
            goto cleanup;
        }
        memset(matrices[m].mat, 0, entries * sizeof(double));
        if (m > H)
            continue;
        for (int col = 0; col < d[m]->nt; ++col) {
            for (int row = 0; row < d[m]->mt; ++row) {
                double *tile = tile_at(d[m], row, col);
                for (int j = 0; j < nb && col * nb + j < n; ++j) {
                    for (int i = 0; i < nb && row * nb + i < n; ++i) {
                        const int r = row * nb + i, c = col * nb + j;
                        double value = initial_value(m, r, c, n);
                        tile[i + (size_t)j * nb] = value;
                        if (verify)
                            ref[m][r + (size_t)c * n] = value;
                    }
                }
            }
        }
    }

    printf("PTG FP64 LU n=%d nb=%d cores=%d max_rounds=%d verify=%d\n",
           n, nb, cores, iterations, verify);
    for (int iteration = 1; iteration <= iterations; ++iteration) {
        double norm = NAN;
        struct round_times times = {0};
        const double start = seconds();
        int info = doubling(context, d, &norm, &times);
        const double elapsed = seconds() - start;
        if (info != 0 || !isfinite(norm)) {
            fprintf(stderr, "Doubling failed in round %d (LU info=%d, norm=%g)\n",
                    iteration, info, norm);
            goto cleanup;
        }
        printf("round=%d seconds=%.6f factor_s=%.6f update_h_s=%.6f update_ag_s=%.6f norm=%.9e",
               iteration, elapsed, times.factor, times.update_h, times.update_ag, norm);
        if (verify) {
            double ref_norm;
            if (reference(n, ref, &ref_norm, pivots) != 0) {
                fputs("\nLAPACK reference failed\n", stderr);
                goto cleanup;
            }
            const double norm_error = fabs(norm - ref_norm) / fmax(ref_norm, 1e-300);
            const double a_error = relative_error(d[A], ref[A]);
            const double g_error = relative_error(d[G], ref[G]);
            const double h_error = relative_error(d[H], ref[H]);
            printf(" rel_A=%.3e rel_G=%.3e rel_H=%.3e rel_norm=%.3e",
                   a_error, g_error, h_error, norm_error);
            if (!(a_error <= 1e-12 && g_error <= 1e-12 && h_error <= 1e-12 &&
                  norm_error <= 1e-12 && (norm < 1e-15) == (ref_norm < 1e-15))) {
                fputs("\nPTG and LAPACK reference diverged\n", stderr);
                goto cleanup;
            }
        }
        puts("");
        fflush(stdout);
        if (norm < 1e-15)
            break;
    }
    status = EXIT_SUCCESS;

cleanup:
#if defined(DPLASMA_HAVE_CUDA)
    if (cuda_info != PARSEC_INFO_ID_UNDEFINED)
        parsec_info_unregister(&parsec_per_stream_infos, cuda_info, NULL);
#endif
    /* Release the descriptors' GPU copies while the device modules still exist. */
    for (int m = 0; m < descriptors; ++m) {
        parsec_tiled_matrix_destroy(&matrices[m].super);
        if (matrices[m].mat != NULL)
            parsec_data_free(matrices[m].mat);
        free(ref[m]);
    }
    free(pivots);
#if defined(DPLASMA_HAVE_CUDA)
    /* Drain cached copies before PaRSEC tears down the data and device pools. */
    if (cuda_info != PARSEC_INFO_ID_UNDEFINED)
        parsec_devices_release_memory();
#endif
    parsec_fini(&context);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif
    return status;
}

// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <Eigen/Dense>
#include "flops.h"
#include "cusolver_util.h"
#include "cnpy.h"

using std::cout;
using std::endl;
using std::vector;
using namespace Eigen;

void genuniform(curandGenerator_t rng, float *A, size_t num)
{
    curandGenerateUniform(rng, A, num);
}

void genuniform(curandGenerator_t rng, double *A, size_t num)
{
    curandGenerateUniformDouble(rng, A, num);
}

template <typename T>
void GPU_fill_rand(T *A, int nr_rows_A, int nr_cols_A)
{
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    // Fill the array with random numbers on the device
    genuniform(prng, A, nr_rows_A * nr_cols_A);
}
// void getrf_test(size_t m, size_t k)
// {
//     cusolverDnHandle_t cusolverH = NULL;
//     cudaStream_t stream = NULL;

//     using data_type = double;

//     // const int64_t m = 3;
//     const int64_t lda = m;
//     const int64_t ldb = m;
//     /*       | 1 2 3  |
//      *   A = | 4 5 6  |
//      *       | 7 8 10 |
//      *
//      * without pivoting: A = L*U
//      *       | 1 0 0 |      | 1  2  3 |
//      *   L = | 4 1 0 |, U = | 0 -3 -6 |
//      *       | 7 2 1 |      | 0  0  1 |
//      *
//      * with pivoting: P*A = L*U
//      *       | 0 0 1 |
//      *   P = | 1 0 0 |
//      *       | 0 1 0 |
//      *
//      *       | 1       0     0 |      | 7  8       10     |
//      *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
//      *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
//      *
//      * (LU) x = B
//      */

//     // std::vector<data_type> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
//     // std::vector<data_type> B = {1.0, 2.0, 3.0};
//     std::vector<data_type> A(m*m,0);
//     std::vector<data_type> B(m*k,0);
//     for(size_t i=0; i<m*m; i++) A[i] = (double)(rand())/RAND_MAX;
//     for(size_t i=0; i<m; i++) A[i*m+i] += 3.0;
//     for(size_t i=0; i<m*k; i++) B[i] = (double)(rand())/RAND_MAX;
//     std::vector<data_type> X(m*k,0);
//     std::vector<data_type> LU(lda * m, 0);
//     std::vector<int64_t> Ipiv(m, 0);
//     int info = 0;

//     data_type *d_A = nullptr;  /* device copy of A */
//     data_type *d_B = nullptr;  /* device copy of B */
//     int64_t *d_Ipiv = nullptr; /* pivoting sequence */
//     int *d_info = nullptr;     /* error info */

//     size_t d_lwork = 0;     /* size of workspace */
//     void *d_work = nullptr; /* device workspace for getrf */
//     size_t h_lwork = 0;     /* size of workspace */
//     void *h_work = nullptr; /* host workspace for getrf */

//     const int pivot_on = 1;
//     const int algo = 0;

//     if (pivot_on)
//     {
//         std::printf("pivot is on : compute P*A = L*U \n");
//     }
//     else
//     {
//         std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
//     }

//     std::printf("A = (matlab base-1)\n");
//     print_matrix(m, m, A.data(), lda);
//     std::printf("=====\n");

//     std::printf("B = (matlab base-1)\n");
//     print_matrix(m, 1, B.data(), ldb);
//     std::printf("=====\n");

//     /* step 1: create cusolver handle, bind a stream */
//     CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

//     CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//     CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

//     /* Create advanced params */
//     cusolverDnParams_t params;
//     CUSOLVER_CHECK(cusolverDnCreateParams(&params));
//     if (algo == 0)
//     {
//         std::printf("Using New Algo\n");
//         CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
//     }
//     else
//     {
//         std::printf("Using Legacy Algo\n");
//         CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
//     }

//     /* step 2: copy A to device */
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

//     CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
//                                stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
//                                stream));

//     /* step 3: query working space of getrf */
//     CUSOLVER_CHECK(
//         cusolverDnXgetrf_bufferSize(cusolverH, params, m, m, traits<data_type>::cuda_data_type, d_A,
//                                     lda, traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

//     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

//     /* step 4: LU factorization */
//     if (pivot_on)
//     {
//         CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<data_type>::cuda_data_type,
//                                         d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
//                                         d_lwork, h_work, h_lwork, d_info));
//     }
//     else
//     {
//         CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<data_type>::cuda_data_type,
//                                         d_A, lda, nullptr, traits<data_type>::cuda_data_type,
//                                         d_work, d_lwork, h_work, h_lwork, d_info));
//     }

//     if (pivot_on)
//     {
//         CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int64_t) * Ipiv.size(),
//                                    cudaMemcpyDeviceToHost, stream));
//     }
//     CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
//                                stream));
//     CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     std::printf("after Xgetrf: info = %d\n", info);
//     if (0 > info)
//     {
//         std::printf("%d-th parameter is wrong \n", -info);
//         exit(1);
//     }
//     if (pivot_on)
//     {
//         std::printf("pivoting sequence, matlab base-1\n");
//         for (int j = 0; j < m; j++)
//         {
//             std::printf("Ipiv(%d) = %lu\n", j + 1, Ipiv[j]);
//         }
//     }
//     std::printf("L and U = (matlab base-1)\n");
//     print_matrix(m, m, LU.data(), lda);
//     std::printf("=====\n");

//     /*
//      * step 5: solve A*X = B
//      *       | 1 |       | -0.3333 |
//      *   B = | 2 |,  X = |  0.6667 |
//      *       | 3 |       |  0      |
//      *
//      */
//     if (pivot_on)
//     {
//         CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, k, /* nrhs */
//                                         traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
//                                         traits<data_type>::cuda_data_type, d_B, ldb, d_info));
//     }
//     else
//     {
//         CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, k, /* nrhs */
//                                         traits<data_type>::cuda_data_type, d_A, lda, nullptr,
//                                         traits<data_type>::cuda_data_type, d_B, ldb, d_info));
//     }

//     CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(), cudaMemcpyDeviceToHost,stream));
//     CUDA_CHECK(cudaStreamSynchronize(stream));

//     std::printf("X = (matlab base-1)\n");
//     print_matrix(m, k, X.data(), ldb);
//     std::printf("=====\n");
//     std::vector<int64_t> recoverP(m, 0);
//     for (int i = 0; i < m; i++)
//         recoverP[i] = i;
//     cout << m << endl;
//     cout << Ipiv.size() << endl;
//     for (int i = 0; i < Ipiv.size(); i++)
//     {
//         int tmp = recoverP[i];
//         recoverP[i] = recoverP[Ipiv[i] - 1];
//         recoverP[Ipiv[i] - 1] = tmp;
//     }

//     auto Pmat = Matrix<double, Dynamic, Dynamic>(m, m);
//     for (size_t i = 0; i < m; i++)
//         Pmat(i, recoverP[i]) = 1.0;

//     auto Amat = Map<Matrix<double, Dynamic, Dynamic>>(A.data(), m, m);
//     auto Xmat = Map<Matrix<double, Dynamic, Dynamic>>(X.data(), m, k);
//     auto Bmat = Map<Matrix<double, Dynamic, Dynamic>>(B.data(), m, k);
//     cout << "Xmat" << Xmat << endl;
//     auto tmpB = Pmat * Amat * Xmat;
//     cout << "=====" << endl;
//     cout << tmpB << endl;
//     cout << "========" << endl;
//     cout << Pmat*Bmat << endl;
//     // cout << Bmat.block(0,0,3,1) << endl;

//     /* free resources */
//     CUDA_CHECK(cudaFree(d_A));
//     CUDA_CHECK(cudaFree(d_B));
//     CUDA_CHECK(cudaFree(d_Ipiv));
//     CUDA_CHECK(cudaFree(d_info));
//     CUDA_CHECK(cudaFree(d_work));
//     CUSOLVER_CHECK(cusolverDnDestroyParams(params));
//     CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
//     CUDA_CHECK(cudaStreamDestroy(stream));
//     CUDA_CHECK(cudaDeviceReset());
// }

template <typename T>
void getrf_test(size_t m, size_t k)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    const int64_t lda = m;
    const int64_t ldb = m;
    std::vector<T> A(m * m, 0);
    std::vector<T> B(m * k, 0);
    for (size_t i = 0; i < m * m; i++)
        A[i] = (double)(rand()) / RAND_MAX;
    for (size_t i = 0; i < m; i++)
        A[i * m + i] += 3.0;
    for (size_t i = 0; i < m * k; i++)
        B[i] = (double)(rand()) / RAND_MAX;
    std::vector<T> X(m * k, 0);
    std::vector<T> LU(lda * m, 0);
    std::vector<int64_t> Ipiv(m, 0);
    int info = 0;
    T *d_A = nullptr;          /* device copy of A */
    T *d_B = nullptr;          /* device copy of B */
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */
    size_t d_lwork = 0;        /* size of workspace */
    void *d_work = nullptr;    /* device workspace for getrf */
    size_t h_lwork = 0;        /* size of workspace */
    void *h_work = nullptr;    /* host workspace for getrf */
    const int pivot_on = 1;
    const int algo = 0;
    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* Create advanced params */
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    if (algo == 0)
    {
        std::printf("Using New Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
    }
    else
    {
        std::printf("Using Legacy Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
    }

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(T) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(T) * B.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(
        cusolverDnXgetrf_bufferSize(cusolverH, params, m, m, traits<T>::cuda_data_type, d_A,
                                    lda, traits<T>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * d_lwork));

    // add timing
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    /* step 4: LU factorization */
    if (pivot_on)
    {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<T>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<T>::cuda_data_type, d_work,
                                        d_lwork, h_work, h_lwork, d_info));
    }
    else
    {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<T>::cuda_data_type,
                                        d_A, lda, nullptr, traits<T>::cuda_data_type,
                                        d_work, d_lwork, h_work, h_lwork, d_info));
    }

    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xgetrf: info = %d\n", info);
    if (0 > info)
    {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on)
    {
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, k, /* nrhs */
                                        traits<T>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<T>::cuda_data_type, d_B, ldb, d_info));
    }
    else
    {
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, k, /* nrhs */
                                        traits<T>::cuda_data_type, d_A, lda, nullptr,
                                        traits<T>::cuda_data_type, d_B, ldb, d_info));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    rawtime.push_back(milliseconds * 1e-3);
    cout << (FLOPS_DGETRF(m, m) + FLOPS_DGETRS(m, m)) / rawtime[0] * 1e-9 << " GFLOPS/s" << endl;

    if (pivot_on)
    {
        CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int64_t) * Ipiv.size(), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(T) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(T) * X.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int64_t> recoverP(m, 0);
    for (int i = 0; i < m; i++)
        recoverP[i] = i;
    for (int i = 0; i < Ipiv.size(); i++)
    {
        int tmp = recoverP[i];
        recoverP[i] = recoverP[Ipiv[i] - 1];
        recoverP[Ipiv[i] - 1] = tmp;
    }

    auto Pmat = Matrix<double, Dynamic, Dynamic>(m, m);
    for (size_t i = 0; i < m; i++)
        Pmat(i, recoverP[i]) = 1.0;

    // auto Amat = Map<Matrix<double, Dynamic, Dynamic>>(A.data(), m, m);
    // auto Xmat = Map<Matrix<double, Dynamic, Dynamic>>(X.data(), m, k);
    // auto Bmat = Map<Matrix<double, Dynamic, Dynamic>>(B.data(), m, k);
    // auto tmpB = Pmat * Amat * Xmat;
    // cout << "=====" << endl;
    // cout << tmpB.block(0, 0, 10, 1) << endl;
    // cout << "========" << endl;
    // cout << (Pmat * Bmat).block(0, 0, 10, 1) << endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
}


int main(int argc, char *argv[])
{
    getrf_test<double>(15000, 15000);
    return 0;
}
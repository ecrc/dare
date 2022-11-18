#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <Eigen/Dense>
using std::cout;
using std::endl;
using std::vector;
using namespace Eigen;

void genuniform(curandGenerator_t rng, float * A, size_t num)
{
    curandGenerateUniform(rng, A, num);
}

void genuniform(curandGenerator_t rng, double * A, size_t num)
{
    curandGenerateUniformDouble(rng, A, num);
}

template<typename T>
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

void checkcudastatus(cublasStatus_t st){
    if(st != CUBLAS_STATUS_SUCCESS){
        printf("cuda status failed. \n");
        exit(0);
    }
}

void cugemm(cublasHandle_t handle, double *A, double *B, double *C, int M, int N, int K)
{
    double one = 1.0;
    double zero = 0.0;
    auto rt = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, M, B, K, &zero, C, M);
    checkcudastatus(rt);
}

void cugemm(cublasHandle_t handle, float *A, float *B, float *C, int M, int N, int K)
{
    float one = 1.0;
    float zero = 0.0;
    auto rt = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, M, B, K, &zero, C, M);
    checkcudastatus(rt);
}

template <typename T>
void gemm_test(size_t M, size_t N, size_t K)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    checkcudastatus(stat);
    size_t MK = M * K;
    size_t KN = K * N;
    size_t MN = M * N;
    T *hst_A, *dev_A;
    T *hst_B, *dev_B;
    T *hst_C, *dev_C;
    T *hst_gres;
    cudaMalloc((void **)&dev_A, MK * sizeof(T));
    cudaMalloc((void **)&dev_B, KN * sizeof(T));
    cudaMalloc((void **)&dev_C, MN * sizeof(T));
    GPU_fill_rand(dev_A, M, K);
    GPU_fill_rand(dev_B, K, N);
    hst_A = new T[MK];
    hst_B = new T[KN];
    hst_C = new T[MN];
    hst_gres = new T[MN];
    cudaMemcpy(hst_A, dev_A, MK * sizeof(T), cudaMemcpyDefault);
    cudaMemcpy(hst_B, dev_B, KN * sizeof(T), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for (int i = 0; i < 100; i++)
    {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cugemm(handle, dev_A, dev_B, dev_C, M, N, K);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        rawtime.push_back(milliseconds * 1e-3);
    }
    cudaDeviceSynchronize();
    std::sort(rawtime.begin(), rawtime.end());
    double medtime = rawtime[rawtime.size()/2];
    cout << "median time " << medtime << endl;
    cout << 2 * M * N * K / medtime * 1e-9 << " GFLOPS/s" << endl;
    if (1)
    {
        auto Amat = Map<Matrix<T,Dynamic,Dynamic>>(hst_A, M,K);
        auto Bmat = Map<Matrix<T,Dynamic,Dynamic>>(hst_B, K,N);
        auto Cmat = Map<Matrix<T,Dynamic,Dynamic>>(hst_C, M,N);
        Cmat = Amat * Bmat;
        // cout << Cmat.block(0,0,10,10) << endl;
        cudaMemcpy(hst_gres, dev_C, MN * sizeof(T), cudaMemcpyDefault);
        auto gpuC = Map<Matrix<T,Dynamic,Dynamic>>(hst_gres, M,N);
        double delta = 0.0;
        
        for(int i=0; i<M; i++)
        {
            for(int j=0; j<N; j++)
            {
                double tmpval = Cmat(i,j) - gpuC(i,j);
                delta = fmax(delta, tmpval * tmpval);
            }
        }
        cout << delta << endl;
    }
}

int main()
{
    gemm_test<float>(1000,1000,1000);
    return 0;
}
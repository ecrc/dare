// Replay full dense DARE linear systems on ONE GPU per process.
// FP64 and FP32 use identical no-pivot LU + two TRSM, reusing LU for A and G.
// Disk I/O, host layout conversion, allocation, warmup and reference-error
// diagnostics are excluded from core timings. FP32 correction timing includes
// GPU casts, residual GEMM, norm checks and FP64 updates. Not a 4-GPU DARE test.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)
#define SOLVER(x) do { auto e=(x); if(e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(#x)+": "+std::to_string(e)); } while(0)
using Clock=std::chrono::steady_clock;
template<class F> double timed(F f) {
    CUDA(cudaDeviceSynchronize());
    auto t=Clock::now(); f(); CUDA(cudaDeviceSynchronize());
    return std::chrono::duration<double>(Clock::now()-t).count();
}
template<class T> struct Device {
    T *p=nullptr;
    explicit Device(size_t n) { CUDA(cudaMalloc((void**)&p,n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&)=delete;
};
void pread_exact(int fd, void *p, size_t bytes, off_t offset) {
    auto *c=(char*)p;
    while(bytes) {
        ssize_t got=pread(fd,c,bytes,offset);
        if(got<0 && errno==EINTR) continue;
        if(got<=0) throw std::runtime_error("Incomplete packed file read");
        c+=got; bytes-=got; offset+=got;
    }
}
int load_tiles(const std::string &path, double *device, int expected=0) {
    int fd=open(path.c_str(),O_RDONLY);
    if(fd<0) throw std::runtime_error("Cannot open "+path);
    try {
        char magic[8]; uint64_t h[3];
        pread_exact(fd,magic,8,0); pread_exact(fd,h,24,8);
        if(memcmp(magic,"DARETIL1",8) || h[0]!=h[1] || h[0]==0 || h[0]>46340 || h[2]==0 || h[2]>h[0])
            throw std::runtime_error("Invalid square FP64 tile header: "+path);
        int n=h[0], nb=h[2], full=n/nb, nt=(n+nb-1)/nb;
        if(expected && n!=expected) throw std::runtime_error("Snapshot dimensions disagree");
        struct stat st; if(fstat(fd,&st) || uint64_t(st.st_size)!=32+8*h[0]*h[1])
            throw std::runtime_error("Wrong packed file size: "+path);
        if(!device) { close(fd); return n; }
        std::vector<double> host(size_t(n)*n);
        struct Tile { int i,j,rows,cols; off_t offset; };
        std::vector<Tile> tiles;
        int bounds[]={0,full,nt}; off_t offset=32;
        for(int cr=0;cr<2;++cr) for(int rr=0;rr<2;++rr)
            for(int j=bounds[cr];j<bounds[cr+1];++j) for(int i=bounds[rr];i<bounds[rr+1];++i) {
                int rows=std::min(nb,n-i*nb), cols=std::min(nb,n-j*nb);
                tiles.push_back({i,j,rows,cols,offset}); offset+=size_t(rows)*cols*8;
            }
        std::atomic<size_t> next{0};
        std::exception_ptr errors[4]; std::vector<std::thread> readers;
        for(int worker=0;worker<4;++worker) readers.emplace_back([&,worker] {
            try {
                std::vector<double> tile(size_t(nb)*nb);
                for(size_t k;(k=next.fetch_add(1))<tiles.size();) {
                    auto t=tiles[k]; pread_exact(fd,tile.data(),size_t(t.rows)*t.cols*8,t.offset);
                    for(int col=0;col<t.cols;++col)
                        memcpy(host.data()+size_t(t.j*nb+col)*n+t.i*nb,
                               tile.data()+size_t(col)*t.rows,size_t(t.rows)*8);
                }
            } catch(...) { errors[worker]=std::current_exception(); }
        });
        for(auto &t:readers) t.join();
        for(auto &e:errors) if(e) std::rethrow_exception(e);
        CUDA(cudaMemcpy(device,host.data(),host.size()*8,cudaMemcpyHostToDevice));
        close(fd); return n;
    } catch(...) { close(fd); throw; }
}
template<class To,class From> __global__ void cast_matrix(To *out,const From *in,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) out[i]=To(in[i]);
}
__global__ void add_correction(double *x,const float *delta,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) x[i]+=double(delta[i]);
}
__global__ void difference_squares(const double *x,const double *ref,double *partial,size_t n) {
    __shared__ double sums[256]; double sum=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        double d=x[i]-ref[i]; sum+=d*d;
    }
    sums[threadIdx.x]=sum; __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step) sums[threadIdx.x]+=sums[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) partial[blockIdx.x]=sums[0];
}
double norm(cublasHandle_t h,const double *x,int count) {
    double v; BLAS(cublasDnrm2(h,count,x,1,&v)); return v;
}
double relative_difference(const double *x,const double *ref,Device<double> &partial,size_t count,double refnorm) {
    constexpr int blocks=8192;
    difference_squares<<<blocks,256>>>(x,ref,partial.p,count); CUDA(cudaGetLastError());
    std::vector<double> sums(blocks);
    CUDA(cudaMemcpy(sums.data(),partial.p,blocks*8,cudaMemcpyDeviceToHost));
    long double total=0; for(double s:sums) total+=s;
    return std::sqrt(total)/std::max(refnorm,std::numeric_limits<double>::min());
}
void solve64(cublasHandle_t h,int n,const double *lu,double *b) {
    const double one=1;
    BLAS(cublasDtrsm(h,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT,n,n,&one,lu,n,b,n));
    BLAS(cublasDtrsm(h,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,n,&one,lu,n,b,n));
}
void solve32(cublasHandle_t h,int n,const float *lu,float *b) {
    const float one=1;
    BLAS(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
    BLAS(cublasStrsm(h,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT,n,n,&one,lu,n,b,n));
    BLAS(cublasStrsm(h,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,n,&one,lu,n,b,n));
    BLAS(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));
}
struct Metrics { double backward,relative_residual,xnorm; };
Metrics residual(cublasHandle_t h,int n,const double *s,const double *b,const double *x,double *r,double snorm,double bnorm) {
    double minus=-1,one=1;
    CUDA(cudaMemcpyAsync(r,b,size_t(n)*n*8,cudaMemcpyDeviceToDevice));
    BLAS(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&minus,s,n,x,n,&one,r,n));
    double rn=norm(h,r,n*n), xn=norm(h,x,n*n);
    return {rn/std::max(snorm*xn+bnorm,std::numeric_limits<double>::min()),
            rn/std::max(bnorm,std::numeric_limits<double>::min()),xn};
}
void emit(const std::string &snapshot,const char *rhs,const char *precision,int step,int n,
          double factor,double solve,double check,Metrics m,double difference) {
    std::cout<<snapshot<<','<<rhs<<','<<precision<<','<<step<<','<<n<<','
             <<factor<<','<<solve<<','<<check<<','<<m.backward<<','
             <<m.relative_residual<<','<<difference<<','<<m.xnorm<<'\n'<<std::flush;
}
int main(int argc,char **argv) try {
    if(argc<2 || argc>4) throw std::runtime_error("Usage: benchmark-linear-mixed SNAPSHOT_DIR [DEVICE=0] [MAX_CORRECTIONS=3]");
    std::string dir=argv[1]; int gpu=argc>2?std::stoi(argv[2]):0, maxsteps=argc>3?std::stoi(argv[3]):3;
    if(maxsteps<0 || maxsteps>20) throw std::runtime_error("Invalid correction limit");
    CUDA(cudaSetDevice(gpu));
    cudaDeviceProp props; CUDA(cudaGetDeviceProperties(&props,gpu));
    int n=load_tiles(dir+"/S.tiles",nullptr); size_t count=size_t(n)*n;
    std::cerr<<"one_gpu="<<props.name<<" n="<<n<<" no_pivot=true strsm_math=pedantic snapshot="<<dir<<'\n';
    cublasHandle_t blas; cusolverDnHandle_t solver;
    BLAS(cublasCreate(&blas)); SOLVER(cusolverDnCreate(&solver));
    BLAS(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH));
    Device<double> s(count),lu64(count),b(count),reference(count),x(count),r(count),partial(8192);
    Device<float> lu32(count),rhs32(count); Device<int> info(1);
    load_tiles(dir+"/S.tiles",s.p,n);
    int work64,work32;
    SOLVER(cusolverDnDgetrf_bufferSize(solver,n,n,lu64.p,n,&work64));
    SOLVER(cusolverDnSgetrf_bufferSize(solver,n,n,lu32.p,n,&work32));
    Device<double> workspace(std::max(size_t(work64),(size_t(work32)+1)/2));
    // Warm library startup with a small factor/solve; no large timed call is a first call.
    {
        const int w=std::min(n,32);
        Device<double> warm64(2*w*w); Device<float> warm32(2*w*w);
        std::vector<double> eye(2*w*w,0);
        for(int i=0;i<w;++i) eye[i*w+i]=eye[w*w+i*w+i]=1;
        CUDA(cudaMemcpy(warm64.p,eye.data(),eye.size()*8,cudaMemcpyHostToDevice));
        SOLVER(cusolverDnDgetrf(solver,w,w,warm64.p,w,workspace.p,nullptr,info.p));
        solve64(blas,w,warm64.p,warm64.p+w*w);
        cast_matrix<<<32,256>>>(warm32.p,warm64.p,size_t(2)*w*w);
        SOLVER(cusolverDnSgetrf(solver,w,w,warm32.p,w,(float*)workspace.p,nullptr,info.p));
        solve32(blas,w,warm32.p,warm32.p+w*w);
        const double one=1,zero=0;
        BLAS(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,w,w,w,&one,
                         warm64.p,w,warm64.p,w,&zero,r.p,w));
        (void)norm(blas,r.p,w*w);
        CUDA(cudaDeviceSynchronize());
    }
    double snorm=norm(blas,s.p,int(count));
    int status64=0,status32=0;
    double factor64=timed([&] {
        CUDA(cudaMemcpyAsync(lu64.p,s.p,count*8,cudaMemcpyDeviceToDevice));
        SOLVER(cusolverDnDgetrf(solver,n,n,lu64.p,n,workspace.p,nullptr,info.p));
        CUDA(cudaMemcpy(&status64,info.p,sizeof(int),cudaMemcpyDeviceToHost));
    });
    double factor32=timed([&] {
        cast_matrix<<<8192,256>>>(lu32.p,s.p,count); CUDA(cudaGetLastError());
        SOLVER(cusolverDnSgetrf(solver,n,n,lu32.p,n,(float*)workspace.p,nullptr,info.p));
        CUDA(cudaMemcpy(&status32,info.p,sizeof(int),cudaMemcpyDeviceToHost));
    });
    std::cerr<<"factor_info_fp64="<<status64<<" factor_info_fp32="<<status32<<'\n';
    if(status64) throw std::runtime_error("FP64 LU failed at pivot "+std::to_string(status64));
    std::cout<<std::setprecision(17)<<"snapshot,rhs,precision,corrections,n,factor_s,solve_update_s,checks_s,backward_error_f,relative_residual_f,relative_x_error_f,xnorm_f\n";
    for(const char *rhs:{"A","G"}) {
        load_tiles(dir+"/"+rhs+".tiles",b.p,n); double bnorm=norm(blas,b.p,int(count));
        double solve_time64=timed([&] {
            CUDA(cudaMemcpyAsync(reference.p,b.p,count*8,cudaMemcpyDeviceToDevice));
            solve64(blas,n,lu64.p,reference.p);
        });
        Metrics refmetrics;
        double check64=timed([&] { refmetrics=residual(blas,n,s.p,b.p,reference.p,r.p,snorm,bnorm); });
        emit(dir,rhs,"FP64",0,n,factor64,solve_time64,check64,refmetrics,0);
        if(status32) continue;
        double solve_time32=timed([&] {
            cast_matrix<<<8192,256>>>(rhs32.p,b.p,count);
            solve32(blas,n,lu32.p,rhs32.p);
            cast_matrix<<<8192,256>>>(x.p,rhs32.p,count); CUDA(cudaGetLastError());
        });
        double check32=0;
        for(int step=0;step<=maxsteps;++step) {
            Metrics m;
            check32+=timed([&] { m=residual(blas,n,s.p,b.p,x.p,r.p,snorm,bnorm); });
            double difference=relative_difference(x.p,reference.p,partial,count,refmetrics.xnorm);
            emit(dir,rhs,"FP32_IR",step,n,factor32,solve_time32,check32,m,difference);
            if(!std::isfinite(m.backward) || !std::isfinite(difference) || step==maxsteps) break;
            // Diagnostic sweep: record every finite correction, even after a
            // small backward error. Conditioning can still amplify solution error.
            solve_time32+=timed([&] {
                cast_matrix<<<8192,256>>>(rhs32.p,r.p,count);
                solve32(blas,n,lu32.p,rhs32.p);
                add_correction<<<8192,256>>>(x.p,rhs32.p,count); CUDA(cudaGetLastError());
            });
        }
    }
    SOLVER(cusolverDnDestroy(solver)); BLAS(cublasDestroy(blas)); return 0;
} catch(const std::exception &e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }

// Four independent cold-path MVMs, FP32 throughout. This is an all-inputs-ready
// operator bundle, NOT an assumed AO dependency graph. Matrix dimensions are
// output x input. All four GPUs complete before the wall-clock timer stops.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cublasStatus_t e) { if(e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(e)); }
#define CK(x) check(x)
constexpr int NF=8, NO=4;
#ifndef MVM_ROWS_PER_BLOCK
#define MVM_ROWS_PER_BLOCK 8
#endif
constexpr int ROWS_PER_BLOCK=MVM_ROWS_PER_BLOCK;
static_assert(ROWS_PER_BLOCK==4 || ROWS_PER_BLOCK==8 || ROWS_PER_BLOCK==16,
              "MVM_ROWS_PER_BLOCK must be 4, 8 or 16");
const bool peer_direct=std::getenv("MVM_PEER_DIRECT") && std::string(std::getenv("MVM_PEER_DIRECT"))=="1";
const int NR[NO]={30000,30000,5000,30000}, NC[NO]={30000,5000,30000,20000};
__host__ __device__ float val(uint64_t i) {
    uint32_t x=uint32_t(i)^uint32_t(i>>32); x^=x>>16; x*=0x7feb352dU;
    x^=x>>15; x*=0x846ca68bU; x^=x>>16;
    return (int(x&65535)-32768)/65536.f;
}
struct Op {
    float *a=nullptr,*x=nullptr,*y=nullptr,*tmp=nullptr;
    int rows=0,cols=0,first=0,ld=0,id=0,colfirst=0,globalcols=0;
};
struct Ops { Op o[NO]; };
__global__ void init(Op o) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(o.rows)*o.ld) {
        int r=i/o.ld,c=i%o.ld;
        o.a[i]=c<o.cols ? val(uint64_t(o.first+r)*o.globalcols+o.colfirst+c+uint64_t(o.id)*1000000007ULL):0.f;
    }
}
// Each warp loads aligned float4 vectors for one row. Split-K increases
// parallelism for small output shards; partial sums have a fixed reduction order.
template<int SPLIT>
__global__ void fused_gemv(Ops ops,int frame) {
    int block=blockIdx.x, oi=0;
    for(;oi<NO-1;++oi) {
        int count=((ops.o[oi].rows+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK)*SPLIT;
        if(block<count) break;
        block-=count;
    }
    Op o=ops.o[oi];
    int split=block%SPLIT, row=(block/SPLIT)*ROWS_PER_BLOCK+threadIdx.x/32;
    int lane=threadIdx.x%32;
    if(row>=o.rows) return;
    const float4* a=reinterpret_cast<const float4*>(o.a+size_t(row)*o.ld);
    const float4* x=reinterpret_cast<const float4*>(o.x+size_t(frame)*o.ld);
    float s0=0,s1=0,s2=0,s3=0;
    for(int c=split*32+lane;c<o.ld/4;c+=32*SPLIT) {
        float4 av=a[c], xv=x[c];
        s0=fmaf(av.x,xv.x,s0); s1=fmaf(av.y,xv.y,s1);
        s2=fmaf(av.z,xv.z,s2); s3=fmaf(av.w,xv.w,s3);
    }
    float s=(s0+s1)+(s2+s3);
    for(int d=16;d;d/=2) s+=__shfl_down_sync(0xffffffff,s,d);
    if(lane==0) {
        if constexpr(SPLIT==1) o.y[row]=s;
        else o.tmp[size_t(split)*o.rows+row]=s;
    }
}
template<int SPLIT>
__global__ void finish(Ops ops) {
    int r=blockIdx.x*blockDim.x+threadIdx.x,oi=0;
    for(;oi<NO-1 && r>=ops.o[oi].rows;++oi) r-=ops.o[oi].rows;
    Op o=ops.o[oi]; if(r>=o.rows) return;
    float s=0; for(int k=0;k<SPLIT;++k) s+=o.tmp[size_t(k)*o.rows+r];
    o.y[r]=s;
}
struct GPU {
    Ops ops; cudaStream_t stream{}; cublasHandle_t blas{};
    cudaEvent_t ready{};
    std::array<float*,NO> gathered{},result{};
    std::array<cudaGraphExec_t,NF> graph{};
    std::array<float*,NO> host_x{},host_y{};
    std::array<std::vector<float>,NO> reference;
};
__global__ void reduce_peers(const float* p,float* y,int rows,int peers) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<rows) {float s=0; for(int k=0;k<peers;++k) s+=p[size_t(k)*rows+r]; y[r]=s;}
}
struct PeerOp { const float* p[4]; float* out; int rows; };
struct PeerOps { PeerOp o[NO]; };
__global__ void reduce_direct(PeerOps ops,int peers) {
    int r=blockIdx.x*blockDim.x+threadIdx.x,j=0;
    for(;j<NO-1 && r>=ops.o[j].rows;++j) r-=ops.o[j].rows;
    auto o=ops.o[j]; if(r>=o.rows) return;
    float v=0; for(int k=0;k<peers;++k) v+=o.p[k][r]; o.out[r]=v;
}
void collect(std::vector<GPU>& gs,int cg,bool host) {
    if(cg==1) return;
    for(int d=0;d<int(gs.size());d+=cg) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(int k=0;k<cg;++k) {
            CK(cudaStreamWaitEvent(g.stream,gs[d+k].ready,0));
            if(!peer_direct) for(int j=0;j<NO;++j) {
                auto o=gs[d+k].ops.o[j];
                CK(cudaMemcpyPeerAsync(g.gathered[j]+size_t(k)*o.rows,d,o.y,d+k,o.rows*sizeof(float),g.stream));
            }
        }
        if(peer_direct) {
            PeerOps p{}; int rows=0;
            for(int j=0;j<NO;++j) {
                p.o[j].rows=g.ops.o[j].rows; p.o[j].out=g.result[j]; rows+=p.o[j].rows;
                for(int k=0;k<cg;++k) p.o[j].p[k]=gs[d+k].ops.o[j].y;
            }
            reduce_direct<<<(rows+255)/256,256,0,g.stream>>>(p,cg);
            CK(cudaGetLastError());
        }
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j];
            if(!peer_direct) {
                reduce_peers<<<(o.rows+255)/256,256,0,g.stream>>>(g.gathered[j],g.result[j],o.rows,cg);
                CK(cudaGetLastError());
            }
            if(host) CK(cudaMemcpyAsync(g.host_y[j],g.result[j],o.rows*sizeof(float),cudaMemcpyDeviceToHost,g.stream));
        }
    }
}
template<int S> void kernel(GPU& g,int f) {
    int blocks=0,rows=0;
    for(auto o:g.ops.o) { blocks+=((o.rows+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK)*S; rows+=o.rows; }
    fused_gemv<S><<<blocks,ROWS_PER_BLOCK*32,0,g.stream>>>(g.ops,f);
    if constexpr(S>1) finish<S><<<(rows+255)/256,256,0,g.stream>>>(g.ops);
    CK(cudaGetLastError());
}
void compute(GPU& g,int frame,int split) {
    if(split==0) {
        float one=1,zero=0;
        for(auto o:g.ops.o)
            CK(cublasSgemv(g.blas,CUBLAS_OP_T,o.cols,o.rows,&one,o.a,o.ld,
                          o.x+size_t(frame)*o.ld,1,&zero,o.y,1));
    } else if(split==1) kernel<1>(g,frame);
    else if(split==2) kernel<2>(g,frame);
    else if(split==4) kernel<4>(g,frame);
    else if(split==8) kernel<8>(g,frame);
    else if(split==16) kernel<16>(g,frame);
    else throw std::runtime_error("split must be 0,1,2,4,8,16");
}
void enqueue(GPU& g,int frame,int split,bool host,bool output=true) {
    if(host) for(int j=0;j<NO;++j) {
        auto o=g.ops.o[j];
        CK(cudaMemcpyAsync(o.x+size_t(frame)*o.ld,g.host_x[j]+size_t(frame)*o.ld,
                           o.ld*sizeof(float),cudaMemcpyHostToDevice,g.stream));
    }
    compute(g,frame,split);
    if(host&&output) for(int j=0;j<NO;++j) {
        auto o=g.ops.o[j];
        CK(cudaMemcpyAsync(g.host_y[j],o.y,o.rows*sizeof(float),cudaMemcpyDeviceToHost,g.stream));
    }
}
int main(int argc,char**argv) try {
    if(argc<6) throw std::runtime_error("usage: benchmark-cold-pipeline GPUs samples split(0=cublas) graph(0|1) host(0|1) [raw.csv] [column_groups=1]");
    int ng=std::stoi(argv[1]),samples=std::stoi(argv[2]),split=std::stoi(argv[3]);
    bool graph=std::stoi(argv[4]),host=std::stoi(argv[5]);
    int cg=argc>7?std::stoi(argv[7]):1;
    int count; CK(cudaGetDeviceCount(&count));
    if(ng<1||ng>count||ng>4||samples<1||cg<1||ng%cg) throw std::runtime_error("invalid GPU/sample/distribution count");
    std::vector<GPU> gs(ng);
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,d));
        std::cerr<<"device="<<d<<" name="<<prop.name<<" memory_clock_khz="<<prop.memoryClockRate
                 <<" memory_bus_bits="<<prop.memoryBusWidth<<'\n';
        auto& g=gs[d]; CK(cudaStreamCreateWithFlags(&g.stream,cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&g.ready,cudaEventDisableTiming));
        if(cg>1) for(int k=d/cg*cg;k<(d/cg+1)*cg;++k) if(k!=d) {
            int access=0; CK(cudaDeviceCanAccessPeer(&access,d,k));
            if(!access) throw std::runtime_error("column partitions require peer access");
            auto e=cudaDeviceEnablePeerAccess(k,0);
            if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else CK(e);
        }
        CK(cublasCreate(&g.blas)); CK(cublasSetStream(g.blas,g.stream));
        CK(cublasSetMathMode(g.blas,CUBLAS_PEDANTIC_MATH));
        for(int j=0;j<NO;++j) {
            auto& o=g.ops.o[j]; o.id=j; int rg=ng/cg,rid=d/cg,cid=d%cg;
            o.first=NR[j]*rid/rg; o.rows=NR[j]*(rid+1)/rg-o.first;
            o.globalcols=NC[j]; o.colfirst=NC[j]*cid/cg;
            o.cols=NC[j]*(cid+1)/cg-o.colfirst; o.ld=(o.cols+31)/32*32;
            CK(cudaMalloc(&o.a,size_t(o.rows)*o.ld*sizeof(float)));
            CK(cudaMalloc(&o.x,size_t(NF)*o.ld*sizeof(float)));
            CK(cudaMalloc(&o.y,o.rows*sizeof(float)));
            CK(cudaMalloc(&o.tmp,size_t(16)*o.rows*sizeof(float)));
            if(cg>1 && d%cg==0) {
                CK(cudaMalloc(&g.gathered[j],size_t(cg)*o.rows*sizeof(float)));
                CK(cudaMalloc(&g.result[j],o.rows*sizeof(float)));
            }
            CK(cudaHostAlloc(&g.host_x[j],size_t(NF)*o.ld*sizeof(float),cudaHostAllocPortable));
            CK(cudaHostAlloc(&g.host_y[j],o.rows*sizeof(float),cudaHostAllocPortable));
            for(int f=0;f<NF;++f) for(int c=0;c<o.ld;++c)
                g.host_x[j][size_t(f)*o.ld+c]=c<o.cols?val(987654321ULL+j*100003ULL+f*30007ULL+o.colfirst+c):0.f;
            CK(cudaMemcpyAsync(o.x,g.host_x[j],size_t(NF)*o.ld*sizeof(float),cudaMemcpyHostToDevice,g.stream));
            init<<<(size_t(o.rows)*o.ld+255)/256,256,0,g.stream>>>(o);
            CK(cudaGetLastError());
        }
        CK(cudaStreamSynchronize(g.stream));
    }
    // Validate every output against cuBLAS, and sampled rows independently
    // against CPU FP64 dots of the canonical global matrix and input vectors.
    double worst_l2=0,worst_scaled=0,worst_cpu=0;
    for(int f: {0,NF-1}) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        enqueue(g,f,0,true); CK(cudaStreamSynchronize(g.stream));
        for(int j=0;j<NO;++j) g.reference[j].assign(g.host_y[j],g.host_y[j]+g.ops.o[j].rows);
        enqueue(g,f,split,true); CK(cudaStreamSynchronize(g.stream));
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j]; double diff=0,norm=0;
            for(int r=0;r<o.rows;++r) {
                double ref=g.reference[j][r],v=g.host_y[j][r];
                if(!std::isfinite(v)) throw std::runtime_error("nonfinite output");
                diff+=(v-ref)*(v-ref); norm+=ref*ref;
                worst_scaled=std::max(worst_scaled,std::abs(v-ref)/std::max(1.,std::abs(ref)));
            }
            worst_l2=std::max(worst_l2,std::sqrt(diff/std::max(norm,1e-30)));
            for(int k=0;k<17;++k) {
                int r=(o.rows-1)*k/16; double ref=0;
                for(int c=0;c<o.cols;++c)
                    ref+=double(val(uint64_t(o.first+r)*o.globalcols+o.colfirst+c+uint64_t(j)*1000000007ULL))*g.host_x[j][size_t(f)*o.ld+c];
                worst_cpu=std::max(worst_cpu,std::abs(g.host_y[j][r]-ref)/std::max(1.,std::abs(ref)));
            }
        }
    }
    std::cerr<<"validation relative_l2="<<worst_l2<<" max_scaled="<<worst_scaled<<" cpu_fp64_scaled="<<worst_cpu<<'\n';
    if(worst_l2>5e-6||worst_scaled>1e-4||worst_cpu>1e-4) throw std::runtime_error("validation failed");
    // Make missing host-to-device transfers observable. The host-scoped timed
    // path must restore every frame from its pinned host input, not silently
    // consume the copies used by setup or eager validation.
    if(host) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(auto o:g.ops.o) CK(cudaMemsetAsync(o.x,0,size_t(NF)*o.ld*sizeof(float),g.stream));
        CK(cudaStreamSynchronize(g.stream));
    }
    if(graph) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(int f=0;f<NF;++f) {
            cudaGraph_t cap;
            CK(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
            enqueue(g,f,split,host,cg==1);
            CK(cudaStreamEndCapture(g.stream,&cap));
            CK(cudaGraphInstantiate(&g.graph[f],cap,0));
            CK(cudaGraphDestroy(cap));
        }
    }
    std::vector<double> times; times.reserve(samples);
    std::ofstream raw; if(argc>6) {raw.open(argv[6]); if(!raw) throw std::runtime_error("cannot open raw CSV"); raw<<"frame,latency_us\n";}
    for(int i=-300;i<samples;++i) {
        int f=(i+304)%NF;
        auto start=std::chrono::steady_clock::now();
        for(int d=0;d<ng;++d) {
            CK(cudaSetDevice(d)); auto& g=gs[d];
            if(graph) CK(cudaGraphLaunch(g.graph[f],g.stream));
            else enqueue(g,f,split,host,cg==1);
            if(cg>1) CK(cudaEventRecord(g.ready,g.stream));
        }
        collect(gs,cg,host);
        for(int d=0;d<ng;++d) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(gs[d].stream)); }
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        if(i>=0) { times.push_back(us); if(raw) raw<<i<<','<<std::setprecision(10)<<us<<'\n'; }
    }
    // Check the actual timed path, including graph replay and peer reduction,
    // against a fresh cuBLAS reference for the last frame. Nothing is inferred
    // solely from the pre-capture eager validation.
    int last=(samples-1+304)%NF;
    std::vector<std::array<std::vector<float>,NO>> actual(ng);
    for(int d=0;d<ng;d+=cg) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j]; actual[d][j].resize(o.rows);
            if(host) std::copy(g.host_y[j],g.host_y[j]+o.rows,actual[d][j].begin());
            else CK(cudaMemcpy(actual[d][j].data(),cg==1?o.y:g.result[j],o.rows*sizeof(float),cudaMemcpyDeviceToHost));
        }
    }
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); enqueue(gs[d],last,0,true,false);
        if(cg>1) CK(cudaEventRecord(gs[d].ready,gs[d].stream));
    }
    collect(gs,cg,false);
    double replay_l2=0,replay_cpu=0;
    for(int d=0;d<ng;d+=cg) {
        CK(cudaSetDevice(d)); auto& g=gs[d]; CK(cudaStreamSynchronize(g.stream));
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j]; std::vector<float> ref(o.rows);
            CK(cudaMemcpy(ref.data(),cg==1?o.y:g.result[j],o.rows*sizeof(float),cudaMemcpyDeviceToHost));
            double diff=0,norm=0;
            for(int r=0;r<o.rows;++r) {
                double v=actual[d][j][r]; if(!std::isfinite(v)) throw std::runtime_error("nonfinite replay output");
                diff+=(v-ref[r])*(v-ref[r]); norm+=double(ref[r])*ref[r];
            }
            replay_l2=std::max(replay_l2,std::sqrt(diff/std::max(norm,1e-30)));
            for(int k=0;k<17;++k) {
                int r=(o.rows-1)*k/16; double v=0;
                for(int c=0;c<o.globalcols;++c)
                    v+=double(val(uint64_t(o.first+r)*o.globalcols+c+uint64_t(j)*1000000007ULL))*val(987654321ULL+j*100003ULL+last*30007ULL+c);
                replay_cpu=std::max(replay_cpu,std::abs(actual[d][j][r]-v)/std::max(1.,std::abs(v)));
            }
        }
    }
    std::cerr<<"replay relative_l2="<<replay_l2<<" cpu_fp64_scaled="<<replay_cpu<<'\n';
    if(replay_l2>5e-6||replay_cpu>1e-4) throw std::runtime_error("replay validation failed");
    auto sorted=times; std::sort(sorted.begin(),sorted.end());
    double mean=std::accumulate(times.begin(),times.end(),0.)/samples,var=0;
    for(double t:times) var+=(t-mean)*(t-mean);
    auto pct=[&](double p) {return sorted[size_t(std::ceil(p*samples))-1];};
    int misses=std::count_if(times.begin(),times.end(),[](double t){return t>500;});
    std::cout<<"gpus,samples,split,graph,host,column_groups,peer_direct,min_us,p50_us,p95_us,p99_us,max_us,rms_us,misses_500us,relative_l2,cpu_fp64_scaled,replay_l2,replay_cpu\n"
             <<ng<<','<<samples<<','<<split<<','<<graph<<','<<host<<','<<cg<<','<<peer_direct<<','<<std::fixed<<std::setprecision(3)
             <<sorted.front()<<','<<pct(.5)<<','<<pct(.95)<<','<<pct(.99)<<','<<sorted.back()<<','<<std::sqrt(var/samples)<<','<<misses
             <<','<<std::scientific<<worst_l2<<','<<worst_cpu<<','<<replay_l2<<','<<replay_cpu<<'\n';
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(auto x:g.graph) if(x) CK(cudaGraphExecDestroy(x));
        CK(cublasDestroy(g.blas));
        for(int j=0;j<NO;++j) {auto o=g.ops.o[j]; CK(cudaFree(o.a)); CK(cudaFree(o.x)); CK(cudaFree(o.y)); CK(cudaFree(o.tmp)); CK(cudaFree(g.gathered[j])); CK(cudaFree(g.result[j])); CK(cudaFreeHost(g.host_x[j])); CK(cudaFreeHost(g.host_y[j]));}
        CK(cudaEventDestroy(g.ready));
        CK(cudaStreamDestroy(g.stream));
    }
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }

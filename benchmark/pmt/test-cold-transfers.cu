// Regression for column partitions: suppressing partial-output copies must
// never suppress host-input transfers. Runs on one GPU with tiny matrices.
#ifndef MVM_SOURCE
#define MVM_SOURCE "benchmark-cold-pipeline.cu"
#endif
#define main cold_benchmark_main
#include MVM_SOURCE
#undef main

int main() try {
    CK(cudaSetDevice(0));
    GPU g;
    CK(cudaStreamCreateWithFlags(&g.stream,cudaStreamNonBlocking));
    CK(cublasCreate(&g.blas)); CK(cublasSetStream(g.blas,g.stream));
    CK(cublasSetMathMode(g.blas,CUBLAS_PEDANTIC_MATH));
    for(int j=0;j<NO;++j) {
        auto& o=g.ops.o[j]; o.rows=16; o.cols=o.ld=64;
        CK(cudaMalloc(&o.a,o.rows*o.ld*sizeof(float)));
        CK(cudaMalloc(&o.x,NF*o.ld*sizeof(float)));
        CK(cudaMalloc(&o.y,o.rows*sizeof(float)));
        CK(cudaMalloc(&o.tmp,16*o.rows*sizeof(float)));
        CK(cudaHostAlloc(&g.host_x[j],NF*o.ld*sizeof(float),cudaHostAllocPortable));
        CK(cudaHostAlloc(&g.host_y[j],o.rows*sizeof(float),cudaHostAllocPortable));
        std::vector<float> a(o.rows*o.ld,float(j+1));
        CK(cudaMemcpy(o.a,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
        for(int f=0;f<NF;++f) std::fill(g.host_x[j]+f*o.ld,g.host_x[j]+(f+1)*o.ld,float(f+2));
    }
    for(int split: {0,2}) for(bool graph: {false,true}) for(int f: {0,7}) {
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j]; CK(cudaMemsetAsync(o.x,0,NF*o.ld*sizeof(float),g.stream));
            std::fill(g.host_y[j],g.host_y[j]+o.rows,-123.f);
        }
        CK(cudaStreamSynchronize(g.stream));
        if(graph) {
            cudaGraph_t cap; cudaGraphExec_t exec;
            CK(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
            enqueue(g,f,split,true,false);
            CK(cudaStreamEndCapture(g.stream,&cap));
            CK(cudaGraphInstantiate(&exec,cap,0));
            CK(cudaGraphLaunch(exec,g.stream)); CK(cudaStreamSynchronize(g.stream));
            CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(cap));
        } else {
            enqueue(g,f,split,true,false); CK(cudaStreamSynchronize(g.stream));
        }
        for(int j=0;j<NO;++j) {
            auto o=g.ops.o[j]; std::vector<float> y(o.rows);
            CK(cudaMemcpy(y.data(),o.y,o.rows*sizeof(float),cudaMemcpyDeviceToHost));
            for(int r=0;r<o.rows;++r) {
                if(y[r]!=float(o.cols*(j+1)*(f+2)))
                    throw std::runtime_error("input copy omitted when partial output is disabled");
                if(g.host_y[j][r]!=-123.f)
                    throw std::runtime_error("unexpected partial-output host copy");
            }
        }
    }
    CK(cublasDestroy(g.blas));
    for(int j=0;j<NO;++j) {
        auto o=g.ops.o[j]; CK(cudaFree(o.a)); CK(cudaFree(o.x)); CK(cudaFree(o.y)); CK(cudaFree(o.tmp));
        CK(cudaFreeHost(g.host_x[j])); CK(cudaFreeHost(g.host_y[j]));
    }
    CK(cudaStreamDestroy(g.stream));
    std::cout<<"PASS: input copies preserved, partial-output copies suppressed (eager/graph, cuBLAS/custom, two frames)\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}

// ATA DARE_opt/ataSupervisor.py: x_next=A*x+L*s-G*u_prev; u=K*x_next.
// This tests that dependency graph on synthetic dense FP32 matrices. It does
// not claim to reproduce the separate hot/cold scheduling across camera frames.
#define main independent_mvm_benchmark_main
#include "benchmark-cold-pipeline.cu"
#undef main

struct ControlGPU : GPU {
    float *local=nullptr,*state=nullptr,*command=nullptr;
    cudaEvent_t state_ready{};
};
struct ControlShape {int n=30000,m=20000,u=5000;};
__global__ void init_packed_state(Op o,ControlShape shape) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(o.rows)*o.ld) return;
    int row=o.first+i/o.ld,col=i%o.ld,id=0,cols=shape.n; float sign=1;
    if(col>=o.cols) {o.a[i]=0; return;}
    if(col>=shape.n+shape.m) {col-=shape.n+shape.m; id=1; cols=shape.u; sign=-1;}
    else if(col>=shape.n) {col-=shape.n; id=3; cols=shape.m;}
    o.a[i]=sign*val(uint64_t(row)*cols+col+uint64_t(id)*1000000007ULL);
}
__global__ void combine_state(const float* a,const float* l,const float* g,float* x,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=(a[i]+l[i])-g[i];
}
// One owner computes each state slice. Hybrid K produces full partial command
// vectors; row-distributed K produces disjoint command slices.
struct ControlPeers {
    const float *local[4],*partial[4];
    int state_begin[5],command_begin[5],ng;
    float *state,*command;
};
__global__ void gather_control(ControlPeers p,int n,int u,int stage,int hybrid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(stage!=1 && i<n) {
        int d=0; while(d+1<p.ng && i>=p.state_begin[d+1]) ++d;
        p.state[i]=p.local[d][i-p.state_begin[d]];
        return;
    }
    if(stage==0) return;
    int r=stage==1?i:i-n; if(r<0||r>=u) return;
    float v=0;
    if(hybrid) for(int d=0;d<p.ng;++d) v+=p.partial[d][r];
    else {
        int d=0; while(d+1<p.ng && r>=p.command_begin[d+1]) ++d;
        v=p.partial[d][r-p.command_begin[d]];
    }
    p.command[r]=v;
}
void control_math(ControlGPU& g,Ops ops,int f,int split) {
    if(split==0) {
        float one=1,zero=0;
        for(auto o:ops.o) if(o.rows)
            CK(cublasSgemv(g.blas,CUBLAS_OP_T,o.cols,o.rows,&one,o.a,o.ld,
                          o.x+size_t(f)*o.ld,1,&zero,o.y,1));
    } else {
        GPU view; view.ops=ops; view.stream=g.stream;
        compute(view,f,split);
    }
}
void command_math(ControlGPU& g,int split,bool hybrid) {
    Ops single{}; single.o[0]=g.ops.o[2];
    single.o[0].x=hybrid?g.local:g.state;
    control_math(g,single,0,split);
}
void local_control(ControlGPU& g,int f,int split,bool host,int layout) {
    bool hybrid=layout!=0,packed=layout==2;
    if(host) for(int j: {0,1,3}) {
        if(packed && j!=0) continue;
        auto o=g.ops.o[j];
        CK(cudaMemcpyAsync(o.x+size_t(f)*o.ld,g.host_x[j]+size_t(f)*o.ld,
                           o.ld*sizeof(float),cudaMemcpyHostToDevice,g.stream));
    }
    Ops first=g.ops; first.o[2].rows=0;
    if(packed) first.o[0].y=g.local;
    control_math(g,first,f,split);
    int rows=g.ops.o[0].rows;
    if(!packed) {
        combine_state<<<(rows+255)/256,256,0,g.stream>>>(g.ops.o[0].y,g.ops.o[3].y,g.ops.o[1].y,g.local,rows);
        CK(cudaGetLastError());
    }
    if(hybrid) command_math(g,split,true);
}
ControlPeers peers_for(std::vector<ControlGPU>& gs,int owner,ControlShape shape) {
    ControlPeers p{}; p.ng=gs.size(); p.state=gs[owner].state; p.command=gs[owner].command;
    for(int d=0;d<p.ng;++d) {
        p.local[d]=gs[d].local; p.partial[d]=gs[d].ops.o[2].y;
        p.state_begin[d]=shape.n*d/p.ng; p.command_begin[d]=shape.u*d/p.ng;
    }
    p.state_begin[p.ng]=shape.n; p.command_begin[p.ng]=shape.u;
    return p;
}
void control_step(std::vector<ControlGPU>& gs,ControlShape shape,int f,int split,bool graph,bool host,int layout) {
    bool hybrid=layout!=0;
    int ng=gs.size();
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        if(graph) CK(cudaGraphLaunch(g.graph[f],g.stream));
        else local_control(g,f,split,host,layout);
        CK(cudaEventRecord(hybrid?g.ready:g.state_ready,g.stream));
    }
    if(!hybrid) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(auto& other:gs) CK(cudaStreamWaitEvent(g.stream,other.state_ready,0));
        gather_control<<<(shape.n+255)/256,256,0,g.stream>>>(peers_for(gs,d,shape),shape.n,shape.u,0,0);
        CK(cudaGetLastError()); command_math(g,split,false);
        CK(cudaEventRecord(g.ready,g.stream));
    }
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(auto& other:gs) CK(cudaStreamWaitEvent(g.stream,other.ready,0));
        int length=hybrid?shape.n+shape.u:shape.u;
        gather_control<<<(length+255)/256,256,0,g.stream>>>(peers_for(gs,d,shape),shape.n,shape.u,hybrid?2:1,hybrid);
        CK(cudaGetLastError());
        if(host) {
            auto a=g.ops.o[0];
            CK(cudaMemcpyAsync(g.host_y[0],g.state+a.first,a.rows*sizeof(float),cudaMemcpyDeviceToHost,g.stream));
            if(d==0) CK(cudaMemcpyAsync(g.host_y[2],g.command,shape.u*sizeof(float),cudaMemcpyDeviceToHost,g.stream));
        }
    }
    for(int d=0;d<ng;++d) {CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(gs[d].stream));}
}
struct ControlSnapshot {
    std::vector<std::vector<float>> state,command;
    std::vector<float> host_state,host_command;
};
ControlSnapshot snapshot(std::vector<ControlGPU>& gs,ControlShape shape,bool host) {
    ControlSnapshot s; s.state.resize(gs.size()); s.command.resize(gs.size());
    if(host) {s.host_state.resize(shape.n); s.host_command.assign(gs[0].host_y[2],gs[0].host_y[2]+shape.u);}
    for(int d=0;d<int(gs.size());++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        s.state[d].resize(shape.n); s.command[d].resize(shape.u);
        CK(cudaMemcpy(s.state[d].data(),g.state,shape.n*sizeof(float),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.command[d].data(),g.command,shape.u*sizeof(float),cudaMemcpyDeviceToHost));
        if(host) {auto a=g.ops.o[0]; std::copy(g.host_y[0],g.host_y[0]+a.rows,s.host_state.begin()+a.first);}
    }
    return s;
}
double relative_error(const std::vector<float>& a,const std::vector<float>& b) {
    double diff=0,norm=0;
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("nonfinite controller result");
        diff+=(double(a[i])-b[i])*(double(a[i])-b[i]); norm+=double(b[i])*b[i];
    }
    return std::sqrt(diff/std::max(norm,1e-30));
}
void compare_control(const ControlSnapshot& a,const ControlSnapshot& ref,bool host,double& state_error,double& command_error) {
    for(size_t d=0;d<a.state.size();++d) {
        state_error=std::max(state_error,relative_error(a.state[d],ref.state[0]));
        command_error=std::max(command_error,relative_error(a.command[d],ref.command[0]));
    }
    if(host && (a.host_state!=a.state[0]||a.host_command!=a.command[0]))
        throw std::runtime_error("host output differs from completed device output");
    if(state_error>1e-5||command_error>1e-5) throw std::runtime_error("controller relative L2 check failed");
}
double sampled_control_reference(const ControlSnapshot& s,ControlShape shape,int f) {
    double worst=0;
    for(int k=0;k<17;++k) {
        int row=(shape.n-1)*k/16; double reference=0;
        for(int j: {0,3,1}) {
            int cols=j==0?shape.n:(j==1?shape.u:shape.m); double part=0;
            for(int c=0;c<cols;++c)
                part+=double(val(uint64_t(row)*cols+c+uint64_t(j)*1000000007ULL))*val(987654321ULL+j*100003ULL+f*30007ULL+c);
            reference+=(j==1?-part:part);
        }
        worst=std::max(worst,std::abs(s.state[0][row]-reference)/std::max(1.,std::abs(reference)));
        row=(shape.u-1)*k/16; reference=0;
        for(int c=0;c<shape.n;++c)
            reference+=double(val(uint64_t(row)*shape.n+c+2ULL*1000000007ULL))*s.state[0][c];
        worst=std::max(worst,std::abs(s.command[0][row]-reference)/std::max(1.,std::abs(reference)));
    }
    if(worst>1e-3) throw std::runtime_error("controller CPU FP64 sampled check failed");
    return worst;
}
int main(int argc,char**argv) try {
    if(argc<7) throw std::runtime_error("usage: benchmark-cold-controller GPUs samples split graph host layout(0=rows,1=hybrid,2=packed_hybrid) [raw.csv] [states measurements actuators]");
    int ng=std::stoi(argv[1]),samples=std::stoi(argv[2]),split=std::stoi(argv[3]);
    bool graph=std::stoi(argv[4]),host=std::stoi(argv[5]); int layout=std::stoi(argv[6]);
    bool hybrid=layout!=0,packed=layout==2;
    ControlShape shape; if(argc>10) shape={std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])};
    int count; CK(cudaGetDeviceCount(&count));
    if(ng<1||ng>4||ng>count||samples<1||shape.n<ng||shape.u<ng||shape.m<1||layout<0||layout>2) throw std::runtime_error("invalid controller dimensions, layout or GPU count");
    std::cerr<<"rows_per_block="<<ROWS_PER_BLOCK<<'\n';
    std::vector<ControlGPU> gs(ng);
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin)); auto& g=gs[d];
        cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,d));
        std::cerr<<"device="<<d<<" name="<<prop.name<<" memory_clock_khz="<<prop.memoryClockRate<<" memory_bus_bits="<<prop.memoryBusWidth<<'\n';
        for(int peer=0;peer<ng;++peer) if(peer!=d) {
            int access=0; CK(cudaDeviceCanAccessPeer(&access,d,peer)); if(!access) throw std::runtime_error("controller requires CUDA peer access");
            auto e=cudaDeviceEnablePeerAccess(peer,0); if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else CK(e);
        }
        CK(cudaStreamCreateWithFlags(&g.stream,cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&g.ready,cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&g.state_ready,cudaEventDisableTiming));
        CK(cublasCreate(&g.blas)); CK(cublasSetStream(g.blas,g.stream)); CK(cublasSetMathMode(g.blas,CUBLAS_PEDANTIC_MATH));
        int first=shape.n*d/ng,rows=shape.n*(d+1)/ng-first;
        CK(cudaMalloc(&g.local,size_t((rows+31)/32*32)*sizeof(float)));
        CK(cudaMemset(g.local,0,size_t((rows+31)/32*32)*sizeof(float)));
        // Padded global state is also the K input in the all-row layout.
        CK(cudaMalloc(&g.state,size_t((shape.n+31)/32*32)*sizeof(float)));
        CK(cudaMemset(g.state,0,size_t((shape.n+31)/32*32)*sizeof(float)));
        CK(cudaMalloc(&g.command,shape.u*sizeof(float)));
        for(int j=0;j<NO;++j) {
            if(packed && (j==1||j==3)) continue;
            auto& o=g.ops.o[j]; o.id=j; o.first=first; o.rows=rows;
            o.globalcols=j==1?shape.u:(j==3?shape.m:shape.n); o.cols=o.globalcols;
            if(packed && j==0) o.cols=o.globalcols=shape.n+shape.m+shape.u;
            if(j==2) {
                if(hybrid) {o.first=0; o.rows=shape.u; o.colfirst=first; o.cols=rows;}
                else {o.first=shape.u*d/ng; o.rows=shape.u*(d+1)/ng-o.first;}
            }
            o.ld=(o.cols+31)/32*32;
            CK(cudaMalloc(&o.a,size_t(o.rows)*o.ld*sizeof(float)));
            CK(cudaMalloc(&o.x,size_t(NF)*o.ld*sizeof(float)));
            CK(cudaMalloc(&o.y,o.rows*sizeof(float))); CK(cudaMalloc(&o.tmp,size_t(16)*o.rows*sizeof(float)));
            CK(cudaHostAlloc(&g.host_x[j],size_t(NF)*o.ld*sizeof(float),cudaHostAllocPortable));
            CK(cudaHostAlloc(&g.host_y[j],(j==2?shape.u:o.rows)*sizeof(float),cudaHostAllocPortable));
            for(int f=0;f<NF;++f) for(int c=0;c<o.ld;++c) {
                int id=j,col=o.colfirst+c;
                if(packed && j==0) {
                    if(c>=shape.n+shape.m) {id=1; col=c-shape.n-shape.m;}
                    else if(c>=shape.n) {id=3; col=c-shape.n;}
                }
                g.host_x[j][size_t(f)*o.ld+c]=c<o.cols?val(987654321ULL+id*100003ULL+f*30007ULL+col):0.f;
            }
            CK(cudaMemcpyAsync(o.x,g.host_x[j],size_t(NF)*o.ld*sizeof(float),cudaMemcpyHostToDevice,g.stream));
            if(packed && j==0) init_packed_state<<<(size_t(o.rows)*o.ld+255)/256,256,0,g.stream>>>(o,shape);
            else init<<<(size_t(o.rows)*o.ld+255)/256,256,0,g.stream>>>(o);
            CK(cudaGetLastError());
        }
        CK(cudaStreamSynchronize(g.stream));
    }
    double state_error=0,command_error=0,cpu_error=0;
    for(int f: {0,7}) {
        control_step(gs,shape,f,0,false,true,layout); auto ref=snapshot(gs,shape,true);
        cpu_error=std::max(cpu_error,sampled_control_reference(ref,shape,f));
        control_step(gs,shape,f,split,false,true,layout); auto actual=snapshot(gs,shape,true);
        compare_control(actual,ref,true,state_error,command_error);
        cpu_error=std::max(cpu_error,sampled_control_reference(actual,shape,f));
    }
    if(host) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(int j: {0,1,3}) {auto o=g.ops.o[j]; if(o.rows) CK(cudaMemsetAsync(o.x,0,size_t(NF)*o.ld*sizeof(float),g.stream));}
        CK(cudaStreamSynchronize(g.stream));
    }
    if(graph) for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(int f=0;f<NF;++f) {
            cudaGraph_t cap; CK(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
            local_control(g,f,split,host,layout); CK(cudaStreamEndCapture(g.stream,&cap));
            CK(cudaGraphInstantiate(&g.graph[f],cap,0)); CK(cudaGraphDestroy(cap));
        }
    }
    std::ofstream raw; if(argc>7) {raw.open(argv[7]); if(!raw) throw std::runtime_error("cannot open raw CSV"); raw<<"frame,latency_us\n";}
    std::vector<double> times; times.reserve(samples);
    for(int i=-300;i<samples;++i) {
        int f=(i+304)%NF; auto start=std::chrono::steady_clock::now();
        control_step(gs,shape,f,split,graph,host,layout);
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        if(i>=0) {times.push_back(us); if(raw) raw<<i<<','<<std::setprecision(10)<<us<<'\n';}
    }
    int last=(samples-1+304)%NF; auto actual=snapshot(gs,shape,host);
    control_step(gs,shape,last,0,false,true,layout); auto ref=snapshot(gs,shape,true);
    compare_control(actual,ref,host,state_error,command_error);
    cpu_error=std::max(cpu_error,sampled_control_reference(actual,shape,last));
    std::cerr<<"validated state_l2="<<state_error<<" command_l2="<<command_error<<" cpu_scaled="<<cpu_error<<'\n';
    double mean=std::accumulate(times.begin(),times.end(),0.)/samples,var=0;
    for(double x:times) var+=(x-mean)*(x-mean);
    std::sort(times.begin(),times.end()); auto pct=[&](double p){return times[size_t(std::ceil(p*samples))-1];};
    int misses=std::count_if(times.begin(),times.end(),[](double t){return t>500;});
    std::cout<<"gpus,states,measurements,actuators,samples,split,graph,host,layout,rows_per_block,min_us,p50_us,p95_us,p99_us,max_us,rms_us,misses_500us,state_l2,command_l2,cpu_scaled\n"
             <<ng<<','<<shape.n<<','<<shape.m<<','<<shape.u<<','<<samples<<','<<split<<','<<graph<<','<<host<<','<<layout<<','<<ROWS_PER_BLOCK<<','<<std::fixed<<std::setprecision(3)
             <<times.front()<<','<<pct(.5)<<','<<pct(.95)<<','<<pct(.99)<<','<<times.back()<<','<<std::sqrt(var/samples)<<','<<misses<<','
             <<std::scientific<<state_error<<','<<command_error<<','<<cpu_error<<'\n';
    for(int d=0;d<ng;++d) {
        CK(cudaSetDevice(d)); auto& g=gs[d];
        for(auto graph:g.graph) if(graph) CK(cudaGraphExecDestroy(graph));
        for(int j=0;j<NO;++j) {auto o=g.ops.o[j]; if(o.rows) {CK(cudaFree(o.a)); CK(cudaFree(o.x)); CK(cudaFree(o.y)); CK(cudaFree(o.tmp)); CK(cudaFreeHost(g.host_x[j])); CK(cudaFreeHost(g.host_y[j]));}}
        CK(cudaFree(g.local)); CK(cudaFree(g.state)); CK(cudaFree(g.command)); CK(cublasDestroy(g.blas));
        CK(cudaEventDestroy(g.ready)); CK(cudaEventDestroy(g.state_ready)); CK(cudaStreamDestroy(g.stream));
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}

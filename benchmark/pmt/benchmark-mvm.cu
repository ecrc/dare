// Single-frame dense MVM latency on one or more GPUs with row-sharded output.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Microseconds = std::chrono::duration<double, std::micro>;
constexpr int frame_count = 8;
constexpr int warmup_count = 300;
constexpr double deadline_us = 150.0;

void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cublasGetStatusString(status));
}

#define CHECK(expression) check((expression), #expression)

enum class Timing { stream_events, device_wall, host_roundtrip };

struct Options {
    int rows = 20000;
    int cols = 5000;
    int gpus = 1;
    std::string precision = "fp32";
    std::string layout = "row";
    int samples = 5000;
    int period_us = 0;
};

struct Shard {
    int gpu = 0;
    int row_begin = 0;
    int rows = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cublasHandle_t blas = nullptr;
};

template<class T>
struct ShardBuffers : Shard {
    T *matrix = nullptr;
    T *inputs = nullptr;
    T *output = nullptr;
};

__host__ __device__ double value(uint64_t index)
{
    // Exactly representable at both tested precisions.
    uint32_t x = static_cast<uint32_t>(index) ^ static_cast<uint32_t>(index >> 32);
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return (static_cast<int>(x & 65535U) - 32768) / 65536.0;
}

template<class T>
__global__ void initialize(T *matrix, int row_begin, int rows, int cols, bool row_major)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(rows) * cols)
        return;
    const size_t row = row_major ? i / cols : i % rows;
    const size_t col = row_major ? i % cols : i / rows;
    matrix[i] = static_cast<T>(value((row_begin + row) * cols + col));
}

template<class T>
void gemv(const ShardBuffers<T>& shard, int cols, bool row_major, const T *input)
{
    const auto op = row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int m = row_major ? cols : shard.rows;
    const int n = row_major ? shard.rows : cols;
    const T alpha = 1, beta = 0;
    if constexpr (std::is_same_v<T, float>) {
        CHECK(cublasSgemv(shard.blas, op, m, n, &alpha, shard.matrix, m, input, 1,
                          &beta, shard.output, 1));
    } else {
        CHECK(cublasDgemv(shard.blas, op, m, n, &alpha, shard.matrix, m, input, 1,
                          &beta, shard.output, 1));
    }
}

struct Sample {
    double latency_us;
    double release_late_us;
};

void report(const Options& o, const char *scope, const std::vector<Sample>& samples,
            bool measure_from_release)
{
    std::vector<double> latencies;
    latencies.reserve(samples.size());
    double max_release_late = 0;
    size_t release_misses = 0;
    for (const auto& s : samples) {
        latencies.push_back(s.latency_us);
        max_release_late = std::max(max_release_late, s.release_late_us);
        if (s.latency_us + s.release_late_us > deadline_us)
            ++release_misses;
    }
    std::sort(latencies.begin(), latencies.end());
    const double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double squared_deviation = 0;
    for (double x : latencies)
        squared_deviation += (x - mean) * (x - mean);
    const double rms_jitter = std::sqrt(squared_deviation / latencies.size());
    auto percentile = [&](double p) {
        return latencies[static_cast<size_t>(std::ceil(p * latencies.size())) - 1];
    };
    const auto misses = latencies.end() - std::upper_bound(latencies.begin(), latencies.end(), deadline_us);
    std::cout << o.rows << ',' << o.cols << ',' << o.gpus << ',' << o.precision << ',' << o.layout
              << ',' << o.period_us << ',' << scope << ',' << latencies.size()
              << std::fixed << std::setprecision(3) << ',' << latencies.front() << ',' << mean
              << ',' << percentile(0.5) << ',' << percentile(0.95) << ',' << percentile(0.99)
              << ',' << latencies.back() << ',' << rms_jitter << ',' << misses
              << ',' << 100.0 * misses / latencies.size() << ',' << max_release_late << ',';
    if (measure_from_release)
        std::cout << release_misses;
    std::cout << '\n';
}

template<class T>
void run(const Options& o)
{
    const bool row_major = o.layout == "row";
    const size_t x_bytes = static_cast<size_t>(o.cols) * sizeof(T);
    T *host_inputs = nullptr, *host_output = nullptr;
    CHECK(cudaHostAlloc(&host_inputs, frame_count * x_bytes, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&host_output, static_cast<size_t>(o.rows) * sizeof(T), cudaHostAllocPortable));
    for (size_t i = 0; i < static_cast<size_t>(frame_count) * o.cols; ++i)
        host_inputs[i] = static_cast<T>(value(0x84222325ULL + i));

    std::vector<ShardBuffers<T>> shards(o.gpus);
    for (int gpu = 0; gpu < o.gpus; ++gpu) {
        auto& s = shards[gpu];
        s.gpu = gpu;
        s.row_begin = static_cast<int>(static_cast<int64_t>(o.rows) * gpu / o.gpus);
        s.rows = static_cast<int>(static_cast<int64_t>(o.rows) * (gpu + 1) / o.gpus) - s.row_begin;
        CHECK(cudaSetDevice(gpu));
        CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CHECK(cudaEventCreate(&s.begin));
        CHECK(cudaEventCreate(&s.end));
        CHECK(cublasCreate(&s.blas));
        CHECK(cublasSetStream(s.blas, s.stream));
        CHECK(cublasSetMathMode(s.blas, CUBLAS_PEDANTIC_MATH));
        const size_t elements = static_cast<size_t>(s.rows) * o.cols;
        CHECK(cudaMalloc(&s.matrix, elements * sizeof(T)));
        CHECK(cudaMalloc(&s.inputs, frame_count * x_bytes));
        CHECK(cudaMalloc(&s.output, static_cast<size_t>(s.rows) * sizeof(T)));
        initialize<<<(elements + 255) / 256, 256, 0, s.stream>>>(s.matrix, s.row_begin, s.rows, o.cols, row_major);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpyAsync(s.inputs, host_inputs, frame_count * x_bytes, cudaMemcpyHostToDevice, s.stream));
        CHECK(cudaStreamSynchronize(s.stream));
    }

    auto frame = [&](int index, Timing timing) {
        for (const auto& s : shards) {
            CHECK(cudaSetDevice(s.gpu));
            T *input = s.inputs + static_cast<size_t>(index) * o.cols;
            if (timing == Timing::host_roundtrip)
                CHECK(cudaMemcpyAsync(input, host_inputs + static_cast<size_t>(index) * o.cols,
                                      x_bytes, cudaMemcpyHostToDevice, s.stream));
            if (timing == Timing::stream_events)
                CHECK(cudaEventRecord(s.begin, s.stream));
            gemv(s, o.cols, row_major, input);
            if (timing == Timing::stream_events)
                CHECK(cudaEventRecord(s.end, s.stream));
            if (timing == Timing::host_roundtrip)
                CHECK(cudaMemcpyAsync(host_output + s.row_begin, s.output,
                                      static_cast<size_t>(s.rows) * sizeof(T), cudaMemcpyDeviceToHost, s.stream));
        }
        for (const auto& s : shards) {
            CHECK(cudaSetDevice(s.gpu));
            CHECK(cudaStreamSynchronize(s.stream));
        }
    };

    const int check_frames[] = {0, frame_count - 1};
    for (int index : check_frames) {
        frame(index, Timing::host_roundtrip);
        for (const auto& s : shards) {
            const int check_rows[] = {s.row_begin, s.row_begin + s.rows / 2, s.row_begin + s.rows - 1};
            for (int row : check_rows) {
                double expected = 0;
                for (int col = 0; col < o.cols; ++col)
                    expected += value(static_cast<uint64_t>(row) * o.cols + col) *
                                host_inputs[static_cast<size_t>(index) * o.cols + col];
                const double relative = std::abs(host_output[row] - expected) / std::max(1.0, std::abs(expected));
                const double tolerance = std::is_same_v<T, float> ? 5e-5 : 1e-10;
                if (!std::isfinite(relative) || relative > tolerance)
                    throw std::runtime_error("MVM reference check failed on output row " + std::to_string(row));
            }
        }
    }

    const Timing timings[] = {Timing::stream_events, Timing::device_wall, Timing::host_roundtrip};
    for (const auto timing : timings) {
        const bool events = timing == Timing::stream_events;
        const char *scope = events ? "gpu_stream_max" :
                            (timing == Timing::device_wall ? "device_wall" : "host_roundtrip");
        std::vector<Sample> samples;
        samples.reserve(o.samples);
        auto release = Clock::now();
        const auto period = std::chrono::microseconds(o.period_us);
        const auto spin = std::chrono::microseconds(30);
        for (int i = -warmup_count; i < o.samples; ++i) {
            if (o.period_us > 0) {
                std::this_thread::sleep_until(release - spin);
                while (Clock::now() < release) {}
            }
            const auto start = Clock::now();
            frame((i + warmup_count) % frame_count, timing);
            const auto end = Clock::now();
            if (i >= 0) {
                double latency = Microseconds(end - start).count();
                if (events) {
                    latency = 0;
                    for (const auto& s : shards) {
                        CHECK(cudaSetDevice(s.gpu));
                        float ms = 0;
                        CHECK(cudaEventElapsedTime(&ms, s.begin, s.end));
                        latency = std::max(latency, ms * 1000.0);
                    }
                }
                const double late = o.period_us ? Microseconds(start - release).count() : 0;
                samples.push_back({latency, late});
            }
            if (o.period_us > 0)
                release += period;
        }
        report(o, scope, samples, o.period_us > 0 && !events);
    }

    for (const auto& s : shards) {
        CHECK(cudaSetDevice(s.gpu));
        CHECK(cudaFree(s.output));
        CHECK(cudaFree(s.inputs));
        CHECK(cudaFree(s.matrix));
        CHECK(cublasDestroy(s.blas));
        CHECK(cudaEventDestroy(s.begin));
        CHECK(cudaEventDestroy(s.end));
        CHECK(cudaStreamDestroy(s.stream));
    }
    CHECK(cudaFreeHost(host_output));
    CHECK(cudaFreeHost(host_inputs));
}

int parse_number(const char *arg, int minimum)
{
    size_t used = 0;
    const std::string text(arg);
    const int result = std::stoi(text, &used);
    if (used != text.size() || result < minimum)
        throw std::invalid_argument("invalid numeric argument " + text);
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc > 8 || (argc == 2 && std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [rows=20000] [cols=5000] [gpus=1] [fp32|fp64]"
                     " [row|col] [samples=5000] [period_us=0]\n";
        return argc > 8 ? 2 : 0;
    }
    try {
        Options o;
        if (argc > 1) o.rows = parse_number(argv[1], 1);
        if (argc > 2) o.cols = parse_number(argv[2], 1);
        if (argc > 3) o.gpus = parse_number(argv[3], 1);
        if (argc > 4) o.precision = argv[4];
        if (argc > 5) o.layout = argv[5];
        if (argc > 6) o.samples = parse_number(argv[6], 1);
        if (argc > 7) o.period_us = parse_number(argv[7], 0);
        int devices = 0;
        CHECK(cudaGetDeviceCount(&devices));
        if (o.gpus > devices || o.gpus > o.rows)
            throw std::invalid_argument("requested GPU count exceeds available devices or rows");
        if ((o.precision != "fp32" && o.precision != "fp64") || (o.layout != "row" && o.layout != "col"))
            throw std::invalid_argument("expected fp32|fp64 and row|col");
        if (static_cast<int64_t>(o.rows) * o.cols > std::numeric_limits<int>::max())
            throw std::invalid_argument("matrix too large for this benchmark");
        std::cout << "rows,cols,gpus,precision,layout,period_us,scope,samples,min_us,mean_us,p50_us,p95_us,p99_us,"
                     "max_us,rms_jitter_us,misses_150us,miss_pct,max_release_late_us,release_misses_150us\n";
        if (o.precision == "fp32") run<float>(o);
        else run<double>(o);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

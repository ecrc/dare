// Integration checks for the persistent norm and its selective wait.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include "chameleonsrc/convergence_norm.hpp"

struct Delay { std::atomic<bool> started{false}, finished{false}; };
static void independent_work(void **, void *arg)
{
    auto *delay = static_cast<Delay *>(arg);
    delay->started = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    delay->finished = true;
}

int main(int argc, char **argv)
{
    const int gpus = argc > 1 ? std::atoi(argv[1]) : 0;
    const int init_status = CHAMELEON_Init(2, gpus);
    if (init_status != CHAMELEON_SUCCESS) return 1;
    CHAM_desc_t *a = nullptr;
    constexpr int rows = 37, cols = 29, nb = 16;
    if (CHAMELEON_Desc_Create(&a, CHAMELEON_MAT_ALLOC_TILE, ChamRealDouble,
        nb, nb, nb * nb, rows, cols, 0, 0, rows, cols, 1, 1) != CHAMELEON_SUCCESS) return 1;
    RUNTIME_sequence_t *sequence = nullptr;
    RUNTIME_request_t request = RUNTIME_REQUEST_INITIALIZER;
    CHAMELEON_Sequence_Create(&sequence);
    int failed = 0;
    {
        DareConvergenceNorm reduction(a);
        // Repeated submissions must reinitialize all partial sums. Very small
        // and very large values catch loss of the scaled sum-of-squares rule.
        for (double value : {0., 1., 1e-200, 1e200, 0., -3.}) {
            CHAMELEON_dlaset_Tile_Async(ChamUpperLower, value, value, a, sequence, &request);
            reduction.submit(sequence, &request);
            const double got = reduction.get();
            const double old = CHAMELEON_dlange_Tile(ChamFrobeniusNorm, a);
            const double expected = std::abs(value) * std::sqrt(double(rows * cols));
            const auto close = [](double x, double y) {
                return std::abs(x - y) <= 1e-13 * std::max(std::abs(y), std::numeric_limits<double>::min());
            };
            const bool passed = close(got, expected) && close(got, old);
            if (!passed) ++failed;
            std::cout << "value=" << value << " new=" << got << " old=" << old
                      << " expected=" << expected << " passed=" << passed << '\n';
        }
        CHAMELEON_dlaset_Tile_Async(ChamUpperLower, 1., 1., a, sequence, &request);
        reduction.submit(sequence, &request);
        // Make the scalar ready before placing unrelated work on the CPU.
        // Otherwise a scheduler can queue needed reduction tasks behind the
        // delay, which is legitimate scheduling rather than a global wait.
        CHAMELEON_Sequence_Wait(sequence);
        Delay delay;
        starpu_codelet codelet{};
        codelet.where = STARPU_CPU;
        codelet.cpu_funcs[0] = independent_work;
        auto *task = starpu_task_create();
        task->cl = &codelet;
        task->cl_arg = &delay;
        if (starpu_task_submit(task) != 0) return 1;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!delay.started && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (!delay.started) ++failed;
        const double got = reduction.get();
        std::cout << "independent task finished at norm return: " << delay.finished << '\n';
        if (delay.finished || !std::isfinite(got)) ++failed;
        CHAMELEON_Sequence_Wait(sequence);
        std::cout << "independent task finished after selective norm wait: " << delay.finished << '\n';
    }
    CHAMELEON_Desc_Destroy(&a);
    CHAMELEON_Sequence_Destroy(sequence);
    CHAMELEON_Finalize();
    std::cout << "gpus=" << gpus << " failures=" << failed << '\n';
    return failed ? 1 : 0;
}

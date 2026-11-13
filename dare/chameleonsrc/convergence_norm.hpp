#ifndef DARE_CONVERGENCE_NORM_HPP
#define DARE_CONVERGENCE_NORM_HPP

#include <algorithm>
#include <stdexcept>
#include <chameleon.h>
#include <starpu.h>

extern "C" {
#include <chameleon/tasks_d.h>
CHAM_context_t *chameleon_context_self(void);
}

// Single-rank, persistent equivalent of Chameleon 1.1's Frobenius reduction.
// Reuses its dlaset/dgessq/dplssq/dplssq2 codelets and reduction order.
// submit() does not flush A or wait for all outstanding StarPU tasks.
class DareConvergenceNorm {
public:
    explicit DareConvergenceNorm(CHAM_desc_t *matrix) : matrix_(matrix)
    {
        if (matrix->p != 1 || matrix->q != 1 || CHAMELEON_Comm_size() != 1)
            throw std::runtime_error("DARE_ASYNC_NORM requires one MPI rank");
        const int status = CHAMELEON_Desc_Create(
            &work_, CHAMELEON_MAT_ALLOC_GLOBAL, ChamRealDouble, 2, 1, 2,
            2 * matrix->mt, matrix->nt, 0, 0, 2 * matrix->mt, matrix->nt, 1, 1);
        if (status != CHAMELEON_SUCCESS)
            throw std::runtime_error("Cannot allocate persistent norm workspace");
        result_ = static_cast<starpu_data_handle_t>(RUNTIME_data_getaddr(work_, 0, 0));
    }

    ~DareConvergenceNorm() { CHAMELEON_Desc_Destroy(&work_); }
    DareConvergenceNorm(const DareConvergenceNorm &) = delete;
    DareConvergenceNorm &operator=(const DareConvergenceNorm &) = delete;

    void submit(RUNTIME_sequence_t *sequence, RUNTIME_request_t *request)
    {
        if (pending_) throw std::runtime_error("Norm result has not been consumed");
        if (sequence->status != CHAMELEON_SUCCESS)
            throw std::runtime_error("Cannot submit norm to failed sequence");
        RUNTIME_option_t options;
        auto *context = chameleon_context_self();
        RUNTIME_options_init(&options, context, sequence, request);
        // These four codelets need no worker scratch allocation. The CUDA
        // dgessq implementation manages its own per-worker scratch buffer.
        for (int m = 0; m < matrix_->mt; ++m)
            for (int n = 0; n < matrix_->nt; ++n)
                INSERT_TASK_dlaset(&options, ChamUpperLower, 2, 1, 1.0, 0.0, work_, m, n);
        for (int m = 0; m < matrix_->mt; ++m) {
            const int rows = std::min(matrix_->mb, matrix_->m - m * matrix_->mb);
            for (int n = 0; n < matrix_->nt; ++n) {
                const int cols = std::min(matrix_->nb, matrix_->n - n * matrix_->nb);
                INSERT_TASK_dgessq(&options, ChamEltwise, rows, cols,
                                   matrix_, m, n, work_, m, n);
                if (n > 0)
                    INSERT_TASK_dplssq(&options, ChamEltwise, 1, 1, work_, m, n, work_, m, 0);
            }
        }
        for (int m = 1; m < matrix_->mt; ++m)
            INSERT_TASK_dplssq(&options, ChamEltwise, 1, 1, work_, m, 0, work_, 0, 0);
        INSERT_TASK_dplssq2(&options, 1, work_, 0, 0);
        RUNTIME_options_finalize(&options, context);
        sequence_ = sequence;
        pending_ = true;
    }

    double get()
    {
        if (!pending_) throw std::runtime_error("No norm submission to wait for");
        // Waiting on this one scalar also waits on all its norm dependencies,
        // but leaves independent matrix-update tasks free to continue.
        if (starpu_data_acquire_on_node(result_, STARPU_MAIN_RAM, STARPU_R) != 0)
            throw std::runtime_error("Cannot acquire norm result");
        const auto *data = static_cast<const double *>(
            starpu_data_handle_to_pointer(result_, STARPU_MAIN_RAM));
        const double value = data == nullptr ? -1.0 : data[0];
        starpu_data_release_on_node(result_, STARPU_MAIN_RAM);
        pending_ = false;
        if (data == nullptr || sequence_->status != CHAMELEON_SUCCESS)
            throw std::runtime_error("Norm calculation failed");
        return value;
    }

private:
    CHAM_desc_t *matrix_;
    CHAM_desc_t *work_ = nullptr;
    starpu_data_handle_t result_ = nullptr;
    RUNTIME_sequence_t *sequence_ = nullptr;
    bool pending_ = false;
};

#endif

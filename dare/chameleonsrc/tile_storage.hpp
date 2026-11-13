#ifndef DARE_TILE_STORAGE_HPP
#define DARE_TILE_STORAGE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>

#include <chameleon.h>
#include <starpu.h>

namespace dare_tiles {

constexpr std::size_t header_bytes = 32;
constexpr char magic[] = "DARETIL1";

inline unsigned configured_threads(const char *name)
{
    if (const char *value = std::getenv(name)) {
        if (value[0] < '1' || value[0] > '4' || value[1] != '\0')
            throw std::runtime_error(std::string(name) + " must be 1–4");
        return static_cast<unsigned>(value[0] - '0');
    }
    return 2;
}

inline void check_descriptor(const CHAM_desc_t *desc)
{
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1 ||
        sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559)
        throw std::runtime_error("Packed tiles require little-endian IEEE FP64");
    if (desc->dtyp != ChamRealDouble || desc->p != 1 || desc->q != 1 ||
        desc->i != 0 || desc->j != 0 || desc->m != desc->lm || desc->n != desc->ln ||
        desc->mb != desc->nb || desc->mb <= 0 || desc->m <= 0 || desc->n <= 0 || desc->ooc)
        throw std::runtime_error("Incompatible packed tile descriptor");
}

struct Block {
    int row;
    int col;
    std::size_t offset;
    std::size_t bytes;
    starpu_data_handle_t handle;
};

inline std::vector<Block> blocks(const CHAM_desc_t *desc, bool register_handles)
{
    check_descriptor(desc);
    const int row_bounds[] = {0, desc->m / desc->mb, desc->mt};
    const int col_bounds[] = {0, desc->n / desc->nb, desc->nt};
    const auto elements = std::uint64_t(desc->m) * desc->n;
    if (elements > (std::numeric_limits<off_t>::max() - header_bytes) / sizeof(double) ||
        elements > (std::numeric_limits<std::size_t>::max() - header_bytes) / sizeof(double))
        throw std::runtime_error("Packed tile matrix is too large");

    std::vector<Block> result;
    result.reserve(std::size_t(desc->mt) * desc->nt);
    std::size_t offset = header_bytes;
    for (int col_region = 0; col_region < 2; ++col_region) {
        for (int row_region = 0; row_region < 2; ++row_region) {
            for (int j = col_bounds[col_region]; j < col_bounds[col_region + 1]; ++j) {
                const int width = std::min(desc->nb, desc->n - j * desc->nb);
                for (int i = row_bounds[row_region]; i < row_bounds[row_region + 1]; ++i) {
                    const int height = std::min(desc->mb, desc->m - i * desc->mb);
                    if (desc->get_blkldd(desc, i) != height)
                        throw std::runtime_error("Incompatible packed tile leading dimension");
                    const auto bytes = std::size_t(height) * width * sizeof(double);
                    auto handle = register_handles
                        ? static_cast<starpu_data_handle_t>(RUNTIME_data_getaddr(desc, i, j)) : nullptr;
                    result.push_back({i, j, offset, bytes, handle});
                    offset += bytes;
                }
            }
        }
    }
    if (offset != header_bytes + elements * sizeof(double))
        throw std::runtime_error("Incomplete packed tile descriptor");
    return result;
}

class AcquiredBlock {
public:
    AcquiredBlock(starpu_data_handle_t handle, enum starpu_data_access_mode mode)
        : handle_(handle)
    {
        if (starpu_data_acquire_on_node(handle_, STARPU_MAIN_RAM, mode) != 0)
            throw std::runtime_error("StarPU tile acquisition failed");
    }

    ~AcquiredBlock() { starpu_data_release_on_node(handle_, STARPU_MAIN_RAM); }

    AcquiredBlock(const AcquiredBlock &) = delete;
    AcquiredBlock &operator=(const AcquiredBlock &) = delete;

    char *data() const
    {
        auto *ptr = starpu_data_handle_to_pointer(handle_, STARPU_MAIN_RAM);
        if (ptr == nullptr)
            throw std::runtime_error("StarPU tile has no host buffer");
        return static_cast<char *>(ptr);
    }

private:
    starpu_data_handle_t handle_;
};

template<class Work>
inline void parallel_ranges(std::size_t count, unsigned threads, Work work)
{
    const auto workers = std::min<std::size_t>(threads, count);
    if (workers == 0)
        return;
    const auto run = [&](std::size_t i) {
        const auto start = count * i / workers;
        const auto end = count * (i + 1) / workers;
        work(start, end);
    };
    std::vector<std::future<void>> extra;
    extra.reserve(workers - 1);
    for (std::size_t i = 1; i < workers; ++i)
        extra.push_back(std::async(std::launch::async, run, i));
    run(0);
    for (auto &result : extra)
        result.get();
}

} // namespace dare_tiles

#endif

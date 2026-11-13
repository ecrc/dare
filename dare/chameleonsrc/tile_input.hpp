#ifndef DARE_TILE_INPUT_HPP
#define DARE_TILE_INPUT_HPP

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include "tile_storage.hpp"

namespace dare_tiles {

inline std::uint64_t little_endian_word(const unsigned char *bytes)
{
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= std::uint64_t(bytes[i]) << (8 * i);
    return value;
}

inline void read_exact(int fd, char *destination, std::size_t bytes,
                       off_t offset, const std::string &path)
{
    constexpr std::size_t chunk = 16 * 1024 * 1024;
    while (bytes != 0) {
        const ssize_t n = pread(fd, destination, std::min(bytes, chunk), offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), path);
        }
        if (n == 0)
            throw std::runtime_error(path + ": incomplete tile file");
        bytes -= n;
        destination += n;
        offset += n;
    }
}

inline void read(const std::string &path, CHAM_desc_t *desc, unsigned readers,
                 const std::vector<Block> &tiles)
{
    const auto close_file = [](std::FILE *stream) { std::fclose(stream); };
    std::unique_ptr<std::FILE, decltype(close_file)> file(std::fopen(path.c_str(), "rb"), close_file);
    if (!file)
        throw std::system_error(errno, std::generic_category(), path);
    const int fd = fileno(file.get());
    std::array<unsigned char, header_bytes> header{};
    read_exact(fd, reinterpret_cast<char *>(header.data()), header.size(), 0, path);
    if (std::memcmp(header.data(), magic, 8) != 0)
        throw std::runtime_error(path + ": invalid tile file header");
    const auto rows = little_endian_word(header.data() + 8);
    const auto cols = little_endian_word(header.data() + 16);
    const auto block = little_endian_word(header.data() + 24);
    if (rows != static_cast<unsigned>(desc->m) || cols != static_cast<unsigned>(desc->n) ||
        block != static_cast<unsigned>(desc->mb))
        throw std::runtime_error(path + ": shape or tile size does not match the solver");

    const auto file_size = tiles.back().offset + tiles.back().bytes;
    struct stat info {};
    if (fstat(fd, &info) != 0)
        throw std::system_error(errno, std::generic_category(), path);
    if (!S_ISREG(info.st_mode) || info.st_size != static_cast<off_t>(file_size))
        throw std::runtime_error(path + ": wrong tile file size");

    if (desc->mat == nullptr) {
        parallel_ranges(tiles.size(), readers, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const auto &tile = tiles[i];
                const AcquiredBlock target(tile.handle, STARPU_W);
                read_exact(fd, target.data(), tile.bytes, static_cast<off_t>(tile.offset), path);
            }
        });
        return;
    }

    const auto base = reinterpret_cast<std::uintptr_t>(desc->mat);
    for (const auto &tile : tiles) {
        const auto address = reinterpret_cast<std::uintptr_t>(desc->get_blkaddr(desc, tile.row, tile.col));
        if (address != base + tile.offset - header_bytes)
            throw std::runtime_error(path + ": incompatible contiguous tile layout");
    }
    constexpr std::size_t grain = 8 * 1024 * 1024;
    const auto bytes = file_size - header_bytes;
    const auto chunks = (bytes + grain - 1) / grain;
    parallel_ranges(chunks, readers, [&](std::size_t first, std::size_t last) {
        const auto begin = first * grain;
        const auto end = std::min(bytes, last * grain);
        read_exact(fd, static_cast<char *>(desc->mat) + begin,
                   end - begin, static_cast<off_t>(header_bytes + begin), path);
    });
}

inline void read_inputs(const std::string &directory, int states,
                        const std::array<CHAM_desc_t *, 5> &descs,
                        unsigned readers = configured_threads("DARE_TILE_READERS_PER_FILE"))
{
    if (readers < 1 || readers > 4)
        throw std::runtime_error("Tile readers per file must be 1–4");
    const std::array<const char *, 5> names{"At", "BinvRt", "Btinitial", "Qt", "Rt"};
    // Register StarPU handles on the submission thread before starting the file readers.
    std::array<std::vector<Block>, 5> tiles;
    for (std::size_t i = 0; i < descs.size(); ++i)
        tiles[i] = blocks(descs[i], descs[i]->mat == nullptr);
    std::array<std::future<void>, 5> reads;
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const auto path = directory + "/" + names[i] + std::to_string(states) + ".tiles";
        reads[i] = std::async(std::launch::async, read, path, descs[i], readers, std::cref(tiles[i]));
    }
    for (auto &result : reads)
        result.get();
}

} // namespace dare_tiles

#endif

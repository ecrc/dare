#ifndef DARE_TILE_OUTPUT_HPP
#define DARE_TILE_OUTPUT_HPP

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <system_error>
#include <unistd.h>

#include "tile_storage.hpp"

namespace dare_tiles {

inline void write_exact(int fd, const char *data, std::size_t bytes, off_t offset,
                        const std::string &path)
{
    while (bytes != 0) {
        const ssize_t n = pwrite(fd, data, bytes, offset);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), path);
        }
        if (n == 0)
            throw std::runtime_error(path + ": incomplete tile write");
        bytes -= n;
        data += n;
        offset += n;
    }
}

inline void write(const std::string &path, const CHAM_desc_t *desc,
                  const std::vector<Block> &tiles, unsigned writers)
{
    const auto close_file = [](std::FILE *stream) { std::fclose(stream); };
    std::unique_ptr<std::FILE, decltype(close_file)> file(std::fopen(path.c_str(), "wbx"), close_file);
    if (!file)
        throw std::system_error(errno, std::generic_category(), path);
    const int fd = fileno(file.get());
    const auto file_size = static_cast<off_t>(tiles.back().offset + tiles.back().bytes);
    if (ftruncate(fd, file_size) != 0)
        throw std::system_error(errno, std::generic_category(), path);

    std::array<char, header_bytes> header{};
    std::memcpy(header.data(), magic, 8);
    // The descriptor check guarantees a little-endian host.
    const std::array<std::uint64_t, 3> dimensions{
        std::uint64_t(desc->m), std::uint64_t(desc->n), std::uint64_t(desc->mb)};
    std::memcpy(header.data() + 8, dimensions.data(), dimensions.size() * sizeof(std::uint64_t));
    write_exact(fd, header.data(), header.size(), 0, path);
    parallel_ranges(tiles.size(), writers, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto &tile = tiles[i];
            const AcquiredBlock source(tile.handle, STARPU_R);
            write_exact(fd, source.data(), tile.bytes, static_cast<off_t>(tile.offset), path);
        }
    });
    if (std::fclose(file.release()) != 0)
        throw std::system_error(errno, std::generic_category(), path);
}

inline void write_outputs(const std::string &directory, const CHAM_desc_t *k, const CHAM_desc_t *p)
{
    namespace fs = std::filesystem;
    const auto writers = configured_threads("DARE_TILE_WRITERS_PER_FILE");
    const auto p_tiles = blocks(p, true);
    const auto k_tiles = k == nullptr ? std::vector<Block>{} : blocks(k, true);
    if (fs::exists(directory))
        throw std::runtime_error(directory + ": output generation already exists");

    const std::string prefix = directory + ".tmp.XXXXXX";
    std::vector<char> staging_name(prefix.begin(), prefix.end());
    staging_name.push_back('\0');
    const char *created = mkdtemp(staging_name.data());
    if (created == nullptr)
        throw std::system_error(errno, std::generic_category(), directory);
    const fs::path staging(created);
    try {
        if (k != nullptr) {
            auto gain = std::async(std::launch::async, write, (staging / "K.tiles").string(),
                                   k, std::cref(k_tiles), writers);
            write((staging / "P.tiles").string(), p, p_tiles, writers);
            gain.get();
        } else {
            write((staging / "P.tiles").string(), p, p_tiles, writers);
        }
        fs::rename(staging, directory);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
}

} // namespace dare_tiles

#endif

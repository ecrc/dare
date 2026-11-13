#ifndef DARE_LINEAR_SNAPSHOT_HPP
#define DARE_LINEAR_SNAPSHOT_HPP

#include <chrono>
#include <fstream>
#include <sstream>
#include "tile_output.hpp"

// Diagnostic snapshots of the *unfactored* system and both original RHS.
// Disabled unless DARE_SNAPSHOT_DIR is set. Export perturbs solver timings.
inline void dare_linear_snapshot(int iteration, const CHAM_desc_t *system,
                                 const CHAM_desc_t *alpha, const CHAM_desc_t *beta,
                                 RUNTIME_sequence_t *sequence)
{
    const char *directory = std::getenv("DARE_SNAPSHOT_DIR");
    if (directory == nullptr || directory[0] == '\0') return;
    const char *steps = std::getenv("DARE_SNAPSHOT_STEPS");
    std::istringstream input(steps == nullptr ? "1,8,16" : steps);
    std::string token;
    bool selected = false;
    while (std::getline(input, token, ',')) {
        std::size_t consumed = 0;
        const int step = std::stoi(token, &consumed);
        if (consumed != token.size() || step < 1)
            throw std::runtime_error("Invalid DARE_SNAPSHOT_STEPS");
        selected |= step == iteration;
    }
    if (!selected) return;
    const auto start = std::chrono::steady_clock::now();
    CHAMELEON_Sequence_Wait(sequence);
    if (sequence->status != CHAMELEON_SUCCESS)
        throw std::runtime_error("Cannot snapshot a failed sequence");
    namespace fs = std::filesystem;
    fs::create_directories(directory);
    const fs::path target = fs::path(directory) / ("it-" + std::to_string(iteration));
    if (fs::exists(target)) throw std::runtime_error("Snapshot already exists: " + target.string());
    std::string pattern = target.string() + ".tmp.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char *created = mkdtemp(name.data());
    if (!created) throw std::system_error(errno, std::generic_category(), target.string());
    const fs::path staging(created);
    const auto writers = dare_tiles::configured_threads("DARE_TILE_WRITERS_PER_FILE");
    try {
        const auto s_blocks = dare_tiles::blocks(system, true);
        const auto a_blocks = dare_tiles::blocks(alpha, true);
        const auto g_blocks = dare_tiles::blocks(beta, true);
        auto save = [&](const char *filename, const CHAM_desc_t *desc,
                        const std::vector<dare_tiles::Block> &blocks) {
            dare_tiles::write((staging / filename).string(), desc,
                              blocks, writers);
        };
        auto s = std::async(std::launch::async, save, "S.tiles", system, std::cref(s_blocks));
        auto a = std::async(std::launch::async, save, "A.tiles", alpha, std::cref(a_blocks));
        save("G.tiles", beta, g_blocks);
        s.get();
        a.get();
        std::ofstream manifest(staging / "manifest.json");
        manifest << "{\"iteration\":" << iteration << ",\"n\":" << system->m
                 << ",\"tile\":" << system->mb
                 << ",\"precision\":\"FP64\",\"system\":\"S=I+G*H\","
                    "\"rhs\":[\"A\",\"G\"],\"before_factorization\":true}\n";
        manifest.close();
        if (!manifest) throw std::runtime_error("Cannot write snapshot manifest");
        fs::rename(staging, target);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        throw;
    }
    printf("LINEAR_SNAPSHOT iteration=%d seconds=%.6f path=%s\n", iteration,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
           target.c_str());
    fflush(stdout);
}

#endif

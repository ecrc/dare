#ifndef DARE_FORM_SYSTEM_HPP
#define DARE_FORM_SYSTEM_HPP

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

#include <chameleon.h>

inline bool dare_system_optimization_enabled(const char *name)
{
    const char *value = std::getenv(name);
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

class DareSystemIdentity {
public:
    DareSystemIdentity(CHAM_desc_t *identity, CHAM_desc_t *system, bool diagonal_only)
        : diagonal_only_(diagonal_only)
    {
        if (!diagonal_only_) {
            tiles_.emplace_back(identity, system);
            return;
        }
        for (int offset = 0; offset < system->m; offset += system->mb) {
            const int size = std::min(system->mb, system->m - offset);
            tiles_.emplace_back(
                CHAMELEON_Desc_SubMatrix(identity, offset, offset, size, size),
                CHAMELEON_Desc_SubMatrix(system, offset, offset, size, size));
        }
    }

    DareSystemIdentity(const DareSystemIdentity &) = delete;
    DareSystemIdentity &operator=(const DareSystemIdentity &) = delete;

    ~DareSystemIdentity()
    {
        if (diagonal_only_) {
            // Submatrix views share the parent descriptors' tiles and runtime handles.
            for (const auto &[identity, system] : tiles_) {
                std::free(identity);
                std::free(system);
            }
        }
    }

    void initialize(RUNTIME_sequence_t *sequence, RUNTIME_request_t *request) const
    {
        for (const auto &tile : tiles_) {
            CHAMELEON_dlaset_Tile_Async(ChamUpperLower, 0.0, 1.0, tile.first,
                                      sequence, request);
        }
    }

    void add(RUNTIME_sequence_t *sequence, RUNTIME_request_t *request) const
    {
        for (const auto &[identity, system] : tiles_) {
            CHAMELEON_dgeadd_Tile_Async(ChamNoTrans, 1.0, identity, 1.0, system,
                                      sequence, request);
        }
    }

private:
    bool diagonal_only_;
    std::vector<std::pair<CHAM_desc_t *, CHAM_desc_t *>> tiles_;
};

#endif

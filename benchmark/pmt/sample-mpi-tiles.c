/* Sample packed Chameleon FP64 MPI tiles without changing messages. */
#include <mpi.h>
#include <chameleon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static const char *directory;
static size_t skip_bytes, bytes_seen, every, sequence;
static unsigned saved;
static int rank;

static void initialize(void)
{
    directory = getenv("DARE_TILE_SAMPLE_DIR");
    if (directory == NULL) return;
    const char *skip = getenv("DARE_TILE_SAMPLE_SKIP_GIB");
    const char *stride = getenv("DARE_TILE_SAMPLE_EVERY");
    skip_bytes = (skip == NULL ? 16 : strtoull(skip, NULL, 10)) * (1ULL << 30);
    every = stride == NULL ? 4093 : strtoull(stride, NULL, 10);
    if (every == 0) every = 1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

static void sample_tile(const void *buffer, int count, MPI_Datatype datatype, int dest)
{
    const size_t header = sizeof(size_t) + sizeof(CHAM_tile_t);
    if (count <= 0 || (size_t)count <= header ||
        (datatype != MPI_BYTE && datatype != MPI_PACKED) || dest == MPI_PROC_NULL) return;
    pthread_once(&init_once, initialize);
    if (directory == NULL) return;

    const unsigned char *data = buffer;
    size_t bytes;
    CHAM_tile_t tile;
    memcpy(&bytes, data, sizeof(bytes));
    memcpy(&tile, data + sizeof(bytes), sizeof(tile));
    if (bytes != (size_t)count - header || !(tile.format & CHAMELEON_TILE_FULLRANK) ||
        tile.m <= 0 || tile.n <= 0 ||
        bytes != (size_t)tile.m * tile.n * sizeof(double)) return;

    pthread_mutex_lock(&lock);
    bytes_seen += bytes;
    const size_t index = ++sequence;
    const int keep = bytes_seen >= skip_bytes && index % every == 0 && saved < 32;
    if (keep) ++saved;
    pthread_mutex_unlock(&lock);
    if (!keep) return;

    char path[4096];
    const int length = snprintf(path, sizeof(path), "%s/r%03d-s%09zu-d%03d-%dx%d.f64",
                                directory, rank, index, dest, tile.m, tile.n);
    if (length < 0 || (size_t)length >= sizeof(path)) return;
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        perror("DARE MPI tile sample");
        return;
    }
    const size_t written = fwrite(data + header, 1, bytes, file);
    if (fclose(file) != 0 || written != bytes) {
        fprintf(stderr, "Incomplete DARE MPI tile sample %s\n", path);
        remove(path);
    }
}

int MPI_Isend(const void *buffer, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    sample_tile(buffer, count, datatype, dest);
    return PMPI_Isend(buffer, count, datatype, dest, tag, comm, request);
}

int MPI_Issend(const void *buffer, int count, MPI_Datatype datatype, int dest,
               int tag, MPI_Comm comm, MPI_Request *request)
{
    sample_tile(buffer, count, datatype, dest);
    return PMPI_Issend(buffer, count, datatype, dest, tag, comm, request);
}

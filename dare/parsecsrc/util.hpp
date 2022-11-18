#pragma once
#include <mpi.h>
#define REP(i, n) for (long i = 0; i < (n); i++)
#define ll long long

template <class T>
struct mpi_type_traits
{
    static MPI_Datatype get_type();
};

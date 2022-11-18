#pragma once

#include <vector>
#include <mpi.h>

template <typename T>
void splitdense(int m, int n, int tm, int tn, T * dataptr, T * tileptr);

template <typename T>
void aggregatedense(int m, int n, int tm, int tn, T * dataptr, T * tileptr);

template <typename T>
void senddata(int rank, int m, int n, int tm, int tn, std::vector<std::pair<int,int>> tiles, T *dataptr);

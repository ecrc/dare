// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include "distdata.h"
#include "util.hpp"

#include <iostream>
#include <vector>
using namespace std;

template <typename T>
void splitdense(int m, int n, int tm, int tn, T * dataptr, T * tileptr)
{
    if(tm % m != 0 || tn % n != 0)
    {
        cout << "tile size not matched!" << endl;
    }
    int mb = m/tm, nb = n/tn;
    ll cnt = (ll)0;
    REP(i,mb)
    REP(j,nb)
    {
        REP(row,tm)
        REP(col,tn)
        {
            ll gcnt = (j * tn + col)*m + i*tm + row;
            tileptr[cnt] = dataptr[gcnt];
        }
    }
}


template <> void splitdense(int m, int n, int tm, int tn, float * dataptr, float * tileptr);
template <> void splitdense(int m, int n, int tm, int tn, double * dataptr, double * tileptr);


template <typename T>
void senddata(int rank, int m, int n, int tm, int tn, vector<pair<int,int>> tiles, T *dataptr)
{
    int mb = m/tm, nb = n/tn;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if(rank == 0) 
    {
        cout << "error rank " << endl;
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if(myrank == 0)
    {
        vector<T> sendbuffer(tiles.size() * tm * tn);
        REP(i, tiles.size())
        {
            memcpy(sendbuffer.data()+i*tm*tn,dataptr+(tiles[i].second*mb+tiles[i].first)*tm*tn);
        }
        MPI_Send(sendbuffer.data(), tiles.size()*tm*tn, mpi_type_traits<T>::get_type(), rank, 0, MPI_COMM_WORLD);
    }else if(myrank == rank)
    {
        MPI_Recv(dataptr, tiles.size()*tm*tn, mpi_type_traits<T>::get_type(), 0, 0, MPI_COMM_WORLD);
    }
}

template <> void senddata(int rank, int m, int n, int tm, int tn, vector<pair<int,int>> tiles, float *dataptr);
template <> void senddata(int rank, int m, int n, int tm, int tn, vector<pair<int,int>> tiles, double *dataptr);
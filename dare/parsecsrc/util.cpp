// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include "util.hpp"

#include <complex>
using std::complex;

#define PRIMITIVE(Type, MpiType) template <> MPI_Datatype mpi_type_traits<Type>::get_type() { return MpiType; }
PRIMITIVE(float, MPI_FLOAT);
PRIMITIVE(complex<float>, MPI_COMPLEX);
PRIMITIVE(double, MPI_DOUBLE);
PRIMITIVE(complex<double>, MPI_DOUBLE_COMPLEX);
#undef PRIMITIVE
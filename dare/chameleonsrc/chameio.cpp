// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <iostream>
#include <stdexcept>
#include <chameleon.h>

#include "chameio.hpp"

using std::endl;
using std::cout;

template<typename T>
void read_desc(FitsFile file, CHAM_desc_t *desc, std::string hdu_key ){
    std::vector<T> v;
    if (CHAMELEON_Comm_rank() == 0) {
        v = file.get_data<T>(hdu_key);
        if (v.size() != static_cast<size_t>(desc->lm) * desc->ln) {
            throw std::runtime_error("Incompatible FITS matrix size");
        }
    }
    // The conversion is collective. The column-major buffer lives on rank 0.
    T unused{};
    if (CHAMELEON_Lap2Desc(ChamUpperLower, v.empty() ? &unused : v.data(),
                           desc->lm, desc) != CHAMELEON_SUCCESS) {
        throw std::runtime_error("FITS matrix distribution failed");
    }
}
template void read_desc<float>(FitsFile file, CHAM_desc_t *desc, std::string hdu_key );
template void read_desc<double>(FitsFile file, CHAM_desc_t *desc, std::string hdu_key );


template<typename T>
void read_desc(std::string filename, CHAM_desc_t *desc, std::string hdu_key ){
    ships::io::FitsFile file(filename,"r");
    read_desc<T>(file,desc,hdu_key);
}
template void read_desc<float>(std::string filename, CHAM_desc_t *desc, std::string hdu_key );
template void read_desc<double>(std::string filename, CHAM_desc_t *desc, std::string hdu_key );


template<typename T>
void read_desc(std::string filename, CHAM_desc_t *desc, T *data, std::string hdu_key ){
    ships::io::FitsFile file(filename,"r");
    unsigned long n_elem = desc->lm * desc->ln;
    file.get_data<T>(hdu_key, n_elem, data);
    CHAMELEON_Lap2Desc(ChamUpperLower, data, desc->lm, desc);
}
template void read_desc<float>(std::string filename, CHAM_desc_t *desc, float *data, std::string hdu_key);
template void read_desc<double>(std::string filename, CHAM_desc_t *desc, double *data, std::string hdu_key);


template<typename T>
void write_desc(std::string filename, CHAM_desc_t *desc, std::string hdu_key ){
    const bool root = CHAMELEON_Comm_rank() == 0;
    std::vector<T> v(root ? static_cast<size_t>(desc->lm) * desc->ln : 0);
    T unused{};
    if (CHAMELEON_Desc2Lap(ChamUpperLower, desc, root ? v.data() : &unused,
                           desc->lm) != CHAMELEON_SUCCESS) {
        throw std::runtime_error("FITS matrix collection failed");
    }
    if (root) {
        ships::io::FitsFile file(filename, "w+");
        file.set_data({desc->lm, desc->ln}, v.data(), hdu_key);
    }
}
template void write_desc<float>(std::string filename, CHAM_desc_t *desc, std::string hdu_key );
template void write_desc<double>(std::string filename, CHAM_desc_t *desc, std::string hdu_key );

template<typename T>
void write_desc(std::string filename, CHAM_desc_t *desc, T *data, std::string hdu_key ){
    ships::io::FitsFile file(filename,"w+");

    unsigned long n_elem = desc->lm * desc->ln;
    CHAMELEON_Desc2Lap(ChamUpperLower, desc, data, desc->lm);
    file.set_data({desc->lm,desc->ln},data,hdu_key);
}
template void write_desc(std::string filename, CHAM_desc_t *desc, float *data, std::string hdu_key );
template void write_desc(std::string filename, CHAM_desc_t *desc, double *data, std::string hdu_key );


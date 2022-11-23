// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <iostream>
#include <chameleon.h>

#include "chameio.hpp"

using std::endl;
using std::cout;

template<typename T>
void read_desc(FitsFile file, CHAM_desc_t *desc, std::string hdu_key ){
    std::vector<T> v = file.get_data<T>(hdu_key);
    if(v.size() != desc->lm * desc->ln){
        std::cerr<<"incompatible size"<<std::endl;
    }
    CHAMELEON_Lap2Desc(ChamUpperLower, v.data(), desc->lm, desc);
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
    ships::io::FitsFile file(filename,"w+");
    unsigned long n_elem = desc->lm * desc->ln;
    std::vector<T> v = std::vector<T>(n_elem);
    CHAMELEON_Desc2Lap(ChamUpperLower, desc, v.data(), desc->lm);
    file.set_data({desc->lm,desc->ln},v.data(),hdu_key);
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


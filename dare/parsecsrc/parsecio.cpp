// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include "parsecio.hpp"
#include <iostream>
using std::endl;
using std::cout;

template<typename T>
void read_desc(FitsFile file, parsec_matrix_block_cyclic_t *desc, std::string hdu_key){
    unsigned long n_elem = desc->super.m * desc->super.n;
    file.get_data<T>(hdu_key, n_elem, (T*)desc->mat);
}
template void read_desc<float>(FitsFile file, parsec_matrix_block_cyclic_t *desc, std::string hdu_key);
template void read_desc<double>(FitsFile file, parsec_matrix_block_cyclic_t *desc, std::string hdu_key);

template<typename T>
void write_desc(std::string filename, parsec_matrix_block_cyclic_t *desc, std::string hdu_key){
    ships::io::FitsFile file(filename,"w+");
    unsigned long n_elem = desc->super.m * desc->super.n;
    file.set_data({desc->super.m,desc->super.n},(T*)desc->mat,hdu_key);
}
template void write_desc<float>(std::string filename, parsec_matrix_block_cyclic_t *desc, std::string hdu_key);
template void write_desc<double>(std::string filename, parsec_matrix_block_cyclic_t *desc, std::string hdu_key);
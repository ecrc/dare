#include "fits/types.hpp"
#include "fits/interface.hpp"
#include "fits/read.hpp"
#include "fits/write.hpp"
#include "fits/file.hpp"

#include <iostream>
#include <vector>

using namespace ships::io;


#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

template<typename T>
void read_desc(FitsFile file, parsec_matrix_block_cyclic_t *desc, std::string hdu_key ="PRIMARY");

template<typename T>
void write_desc(std::string filename, parsec_matrix_block_cyclic_t *desc, std::string hdu_key="PRIMARY");


#include "fits/types.hpp"
#include "fits/interface.hpp"
#include "fits/read.hpp"
#include "fits/write.hpp"
#include "fits/file.hpp"

#include <iostream>
#include <vector>

using namespace ships::io;

#include <chameleon.h>

template<typename T>
void read_desc(FitsFile file, CHAM_desc_t *desc, std::string hdu_key ="PRIMARY");

template<typename T>
void read_desc(std::string filename, CHAM_desc_t *desc, T *data, std::string hdu_key="PRIMARY");

template<typename T>
void read_desc(std::string filename, CHAM_desc_t *desc, std::string hdu_key="PRIMARY");

template<typename T>
void write_desc(std::string filename, CHAM_desc_t *desc, std::string hdu_key="PRIMARY");

template<typename T>
void write_desc(std::string filename, CHAM_desc_t *desc, T *data, std::string hdu_key="PRIMARY");

// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <stdexcept>
#include <iostream>
#include <numeric>
#include <fstream>

#include "file.hpp"
#include "interface.hpp"
#include "types.hpp"
#include "read.hpp"
#include "write.hpp"


namespace ships{
namespace io{
#define FITS_MODE_READ 0
#define FITS_MODE_WRITE 1

FitsFile::FitsFile(){
  m_file_name = "";
  m_fits_ptr = nullptr;
  m_n_hdu = 0;
}

FitsFile::FitsFile(const std::string &file_name, const std::string &mode){
  m_fits_ptr = nullptr;
  set(file_name,mode);
}

FitsFile::FitsFile(const FitsFile &file){
  m_file_name = file.m_file_name;
  m_n_hdu = file.m_n_hdu;
  m_mode = file.m_mode;
  m_fits_ptr = fits_open(m_file_name, m_mode);
}

FitsFile::FitsFile(FitsFile &&file){
  m_file_name = file.m_file_name;
  m_n_hdu = file.m_n_hdu;
  m_mode = file.m_mode;
  m_fits_ptr = file.m_fits_ptr;

  file.m_fits_ptr=nullptr;
}

FitsFile::~FitsFile(){
  fits_close(m_fits_ptr,m_n_hdu);
}

void FitsFile::set(const std::string &file_name, const std::string &mode){

  if(m_fits_ptr != nullptr){
    fits_close(m_fits_ptr,m_n_hdu);
    m_fits_ptr = nullptr;
  }

  std::string man = "\nFitsFile mode are :";
    man += "\n\tread      :\"r\"";
    man += "\n\twrite     :\"w\"";
    man += "\n\tappend    :\"a\"";
    man += "\n\toverwrite :\"w+\"";

  int exists = std::ifstream(file_name.c_str()).good();
  if(mode =="a"){
    m_mode = FITS_MODE_WRITE;
  }
  else if(mode == "r"){
    if(!exists){
      throw std::runtime_error(file_name +
        " does not exist.\nCannot open it in 'read' mode."+man);
    }
    m_mode = FITS_MODE_READ;
  }
  else if(mode == "w"){
    if(exists){
      throw std::runtime_error(file_name +
        " already exists.\nCannot open it in 'write' mode."+man);
    }
    m_mode = FITS_MODE_WRITE;
  }
  else if(mode == "w+"){
    if(exists){
      remove(file_name.c_str());
    }
    m_mode = FITS_MODE_WRITE;
  }else{
    throw std::runtime_error("Unknown mode: "+mode+ man);
  }
  m_file_name = file_name;
  m_fits_ptr = fits_open(file_name, m_mode);
  m_n_hdu = fits_nb_hdu(m_fits_ptr);
}


FitsFile& FitsFile::operator=(const FitsFile& file){
  if(*this != file ){
    m_file_name = file.m_file_name;
    m_n_hdu = file.m_n_hdu;
    m_mode = file.m_mode;
    m_fits_ptr = fits_open(m_file_name, m_mode);
  }
  return *this;
}

FitsFile& FitsFile::operator=(FitsFile&& file){
  if(*this != file ){
    m_file_name = file.m_file_name;
    m_n_hdu = file.m_n_hdu;
    m_mode = file.m_mode;
    m_fits_ptr = file.m_fits_ptr;

    file.m_fits_ptr=nullptr;
  }
  return *this;
}

bool FitsFile::operator==(const FitsFile &file){
  return m_fits_ptr == file.m_fits_ptr;
}

bool FitsFile::operator!=(const FitsFile &file){
  return m_fits_ptr != file.m_fits_ptr;
}

std::vector<std::string> FitsFile::hdu_names(){
  std::vector<std::string> names(m_n_hdu);
  for(int i=0; i<m_n_hdu; i++){
    move_to_hdu(i);
    names[i] = current_hdu_name();
  }
  return names;
}

void FitsFile::move_to_hdu(int hdu_index){
  // +1 : rebase indexing to 0
  fits_set_hdu(m_fits_ptr, hdu_index+1);
}

void FitsFile::move_to_hdu(std::string hdu_name){
  if(hdu_name=="PRIMARY"){
    move_to_hdu(0);
  }else{
    fits_set_hdu(m_fits_ptr, hdu_name);
  }
}

std::vector<long> FitsFile::get_data_shape(int hdu_index){
  try{
    move_to_hdu(hdu_index);
    return fits_get_img_shape(m_fits_ptr);
  }catch(const std::exception& e){
    std::cerr<<" in FitsFile "<<m_file_name<<std::endl;
    throw;
  }
}

std::vector<long> FitsFile::get_data_shape(std::string hdu_key){
  try{
    move_to_hdu(hdu_key);
    return fits_get_img_shape(m_fits_ptr);
  }catch(const std::exception& e){
    std::cerr<<" in FitsFile "<<m_file_name<<std::endl;
    throw;
  }
}

unsigned long FitsFile::get_data_size(int hdu_index){
  try{
    std::vector<long> hdu_shape=get_data_shape(hdu_index);
    return std::accumulate(hdu_shape.begin(), hdu_shape.end(), (unsigned long) 1,
      std::multiplies<unsigned long>());
  }catch(const std::exception& e){
    std::cerr<<"Getting number of element in hdu "<< current_hdu()<<std::endl;
    throw;
  }
}

unsigned long FitsFile::get_data_size(std::string hdu_key){
  try{
    std::vector<long> hdu_shape=get_data_shape(hdu_key);
    return std::accumulate(hdu_shape.begin(), hdu_shape.end(), (unsigned long) 1,
      std::multiplies<unsigned long>());
  }catch(const std::exception& e){
    std::cerr<<"Getting number of element in hdu "<< current_hdu()<<std::endl;
    throw;
  }
}

int FitsFile::current_hdu_index(){
  // -1 : rebase indexing to 0
  return fits_current_hdu(m_fits_ptr)-1;
}

std::string FitsFile::current_hdu_name(){
  return fits_get_current_hdu_name(m_fits_ptr);
}

std::string FitsFile::current_hdu(){
    return "#"+std::to_string(current_hdu_index())+" : "+current_hdu_name();
}

template<typename T>
T FitsFile::get_key(int hdu_index, const std::string &key){
  move_to_hdu(hdu_index);
  T value;
  try{
    value = fits_get_key<T>(m_fits_ptr,key);
  }catch(const std::exception& e){
    std::cerr<<" in hdu "<<current_hdu()<<"\n";
    throw;
  }
  return value;
}

int FitsFile::get_type(int hdu_index){
  move_to_hdu(hdu_index);
  return fits_get_img_type(m_fits_ptr);
}

int FitsFile::get_type(const std::string &hdu_key){
  move_to_hdu(hdu_key);
  return fits_get_img_type(m_fits_ptr);
}

template<typename T>
T FitsFile::get_key(const std::string &hdu_key, const std::string &key){
  move_to_hdu(hdu_key);
  return get_key<T>(current_hdu_index(),key);
}
template int FitsFile::get_key(const std::string &hdu_key,
  const std::string &key);
template long FitsFile::get_key(const std::string &hdu_key,
  const std::string &key);
template float FitsFile::get_key( const std::string &hdu_key,
  const std::string &key);
template double FitsFile::get_key(const std::string &hdu_key,
  const std::string &key);
template std::string FitsFile::get_key(const std::string &hdu_key,
  const std::string &key);


template<typename T>
void FitsFile::get_data(int hdu_index, unsigned long n_elem, T *data){
  unsigned long data_size = get_data_size(hdu_index);
  if(n_elem != data_size){
    std::string err =
      "requested number of element does not match content of hdu " +
      current_hdu() + "(requested : "+ std::to_string(n_elem) +
      " , hdu has : " + std::to_string(data_size) + ")";
    throw std::runtime_error(err);
  }
  if(fits_img_type_code<T>() != this->get_type(hdu_index)){
    std::string err = "requested type ("+get_type_name<T>() +
      ") does not match file content ("+
      get_type_of_fits_code(get_type(hdu_index))+")";
    throw std::runtime_error(err);
  }
  try{
    fits_get_data(m_fits_ptr, n_elem, data);
  }catch(const std::exception& e){
    std::cerr<<" in hdu "<<current_hdu()<<"\n";
    throw;
  }
}

template<typename T>
void FitsFile::get_data(const std::string &hdu_key, unsigned long n_elem,
  T *data){
  move_to_hdu(hdu_key);
  get_data(current_hdu_index(),n_elem,data);
}
template
void FitsFile::get_data(const std::string &hdu_key, unsigned long n_elem,
  long *data);
template
void FitsFile::get_data(const std::string &hdu_key, unsigned long n_elem,
  float *data);
template
void FitsFile::get_data(const std::string &hdu_key, unsigned long n_elem,
  double *data);

template<typename T>
std::vector<T> FitsFile::get_data(int hdu_index){
  try{
    unsigned long data_size = get_data_size(hdu_index);
    std::vector<T> data(data_size);
    get_data<T>(hdu_index, data_size, data.data());
    return data;
  }catch(const std::exception& e){
    std::cerr<<"Reading data from hdu "<< current_hdu() <<std::endl;
    throw;
  }
}
template std::vector<long> FitsFile::get_data(int hdu_index);
template std::vector<float> FitsFile::get_data(int hdu_index);
template std::vector<double> FitsFile::get_data(int hdu_index);

template<typename T>
std::vector<T> FitsFile::get_data(const std::string &hdu_key){
  try{
    unsigned long data_size = get_data_size(hdu_key);
    std::vector<T> data(data_size);
    get_data<T>(hdu_key, data_size, data.data());
    return data;
  }catch(const std::exception& e){
    std::cerr<<"Reading data from hdu "<< current_hdu() <<std::endl;
    throw;
  }
}
template std::vector<int> FitsFile::get_data(const std::string &hdu_key);
template std::vector<long> FitsFile::get_data(const std::string &hdu_key);
template std::vector<float> FitsFile::get_data(const std::string &hdu_key);
template std::vector<double> FitsFile::get_data(const std::string &hdu_key);


template<typename T>
void FitsFile::set_key(int hdu_index, const std::string &key, T value,
  const std::string &comment){
  if(m_mode == FITS_MODE_READ){
    throw std::runtime_error(m_file_name +" is open in read only mode");
  }
  move_to_hdu(hdu_index);
  try{
    fits_set_key<T>(m_fits_ptr,key, value, comment);
  }catch(const std::exception& e){
    std::cerr<<" in hdu "<<current_hdu()<<"\n";
    throw;
  }
}

template<typename T>
void FitsFile::set_key(const std::string &hdu_name, const std::string &key,
  T value, const std::string &comment){
  move_to_hdu(hdu_name);
  set_key(current_hdu_index(), key, value, comment);
}
//template void FitsFile::set_key(const std::string &hdu_name,
//  const std::string &key, int value, const std::string &comment);
template void FitsFile::set_key(const std::string &hdu_name,
  const std::string &key, long value, const std::string &comment);
template void FitsFile::set_key(const std::string &hdu_name,
  const std::string &key, float value, const std::string &comment);
template void FitsFile::set_key(const std::string &hdu_name,
  const std::string &key, double value, const std::string &comment);


template<typename T>
void FitsFile::set_data(std::vector<long> shape, const T *data,
  const std::string &hdu_key){
  if(m_mode == FITS_MODE_READ){
    throw std::runtime_error(m_file_name +" is open in read only mode");
  }
  try{
    fits_set_data(m_fits_ptr, shape, data, hdu_key);
  }catch(const std::exception& e){
    std::cerr<<" in hdu "<<current_hdu()<<"\n";
    throw;
  }
  m_n_hdu +=1;
}

template void FitsFile::set_data(std::vector<long> shape, const int *data,
  const std::string &hdu_key);
template void FitsFile::set_data(std::vector<long> shape, const long *data,
  const std::string &hdu_key);
template void FitsFile::set_data(std::vector<long> shape, const float *data,
  const std::string &hdu_key);
template void FitsFile::set_data(std::vector<long> shape, const double *data,
  const std::string &hdu_key);

} // io
} // ships

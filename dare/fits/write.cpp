#include <numeric>
#include <stdexcept>

#include "write.hpp"
#include "interface.hpp"
#include "types.hpp"

namespace ships{
namespace io{
template<typename T>
void fits_set_key(fitsfile *fits_file_ptr, const std::string &key, T value,
  const std::string &comment){
  int status=0;
  const char *fits_comment= (comment=="")?nullptr:comment.c_str();
  if( fits_write_key(fits_file_ptr, fits_key_type_code<T>(), key.c_str(),
    &value, fits_comment, &status)){
    fits_report_error(stderr,status);
    std::string err = "Error while writing the key '"+key+"'";
    throw std::runtime_error(err);
  }
}

template void fits_set_key(fitsfile *fits_file_ptr, const std::string &key,
  int value, const std::string &comment);
template void fits_set_key(fitsfile *fits_file_ptr, const std::string &key,
  long value, const std::string &comment);
template void fits_set_key(fitsfile *fits_file_ptr, const std::string &key,
  float value, const std::string &comment);
template void fits_set_key(fitsfile *fits_file_ptr, const std::string &key,
  double value, const std::string &comment);

void fits_set_key(fitsfile *fits_file_ptr, const std::string &key,
  std::string value, const std::string &comment){
  int status=0;
  const char *fits_comment= (comment=="")?nullptr:comment.c_str();
  char *value_c_str = const_cast<char*>(value.c_str());
  if(fits_write_key(fits_file_ptr, fits_key_type_code<std::string>(),
    key.c_str(), value_c_str, fits_comment, &status)){
    fits_report_error(stderr,status);
    std::string err = "Error while writing the key '"+key+"'";
    throw std::runtime_error(err);
  }
}


template<typename T>
void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> shape,
  const T *data, const std::string &hdu_key){

  int status=0;
  int fpixel = 1; /* first pixel to write      */
  long n_elem = std::accumulate(shape.begin(), shape.end(), (long)1,
    std::multiplies<long>());

  if(fits_create_img(fits_file_ptr, fits_img_type_code<T>(), shape.size(),
    shape.data(), &status)){
      fits_report_error(stderr, status);
      throw std::runtime_error("Error while creating image in hdu #"+
          fits_current_hdu(fits_file_ptr));
  }
  if(fits_write_img(fits_file_ptr, fits_key_type_code<T>(), fpixel, n_elem,
    const_cast<T*>(data), &status)){
      fits_report_error(stderr, status);
      throw std::runtime_error("Error while writing in hdu #"+
          fits_current_hdu(fits_file_ptr));
  }
  if(hdu_key != ""){
    fits_set_key(fits_file_ptr, "EXTNAME", hdu_key, "");
  }
}




template void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> shape, const int *data, const std::string &hdu_key);
template void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> shape, const long *data, const std::string &hdu_key);
template void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> shape, const float *data, const std::string &hdu_key);
template void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> shape, const double *data, const std::string &hdu_key);

} // io
} // ships
#include <stdexcept>
#include <fstream>
#include <stdio.h>
#include <iostream>

#include "interface.hpp"

namespace ships{
namespace io{
fitsfile* fits_open(const std::string &file_name, int mode){
  fitsfile *fits_file_ptr;
  int status=0;
  int exists = std::ifstream(file_name.c_str()).good();

  if(exists){
    if(fits_open_file(&fits_file_ptr, file_name.c_str(), mode, &status)){
      fits_report_error(stderr,status);
      throw std::runtime_error( "Could not open file: '" + file_name+"'" );
    }
  }else{
    if ( fits_create_file(&fits_file_ptr, file_name.c_str(), &status) ){
      fits_report_error(stderr, status);
    }
  }
  return fits_file_ptr;
}

void fits_close(fitsfile *fits_file_ptr, int n_hdu){
  int status=0;
  if(fits_file_ptr){
    //fits_file_name(fits_file_ptr, file_name, &status);
    std::string file_name = fits_file_name(fits_file_ptr);
    bool last = (fits_file_ptr->Fptr->open_count == 1);
    if(fits_close_file(fits_file_ptr,&status)){
      if(status == UNKNOWN_REC && n_hdu<1){
        //empty file
        std::cerr<<"Empty fits file : "<< file_name << " was removed"<<std::endl;
        remove(file_name.c_str());
      }else{
        fits_report_error(stderr,status);
        throw std::runtime_error("Error while closing file.");
      }
    }
    if(last){
      fits_file_ptr = nullptr;
    }
  }
}

int fits_nb_hdu(fitsfile *fits_file_ptr){
    int n;
    int status=0;
    if(fits_get_num_hdus(fits_file_ptr, &n, &status)){
        fits_report_error(stderr,status);
        throw std::runtime_error("Error getting the number of hdu.");
    }
    return n;
}

int fits_current_hdu(fitsfile *fits_file_ptr){
    int n;
    fits_get_hdu_num(fits_file_ptr, &n);
    return n;
}

void fits_set_hdu(fitsfile *fits_file_ptr, int hdu_index){
    int status=0;
    if(fits_movabs_hdu(fits_file_ptr, hdu_index, nullptr, &status)){
        fits_report_error(stderr,status);
        throw std::runtime_error("Error setting the current hdu to " +
            std::to_string(hdu_index));
    }
}

void fits_set_hdu(fitsfile *fits_file_ptr, const std::string &hdu_name){
    int status=0;
    if(fits_movnam_hdu(fits_file_ptr, 0,
                    const_cast<char*>(hdu_name.c_str()), 0, &status)){
        fits_report_error(stderr,status);
        throw std::runtime_error("Error setting the current hdu to " +
            hdu_name);
    }
}

std::string fits_file_name(fitsfile *fits_file_ptr){
  int status=0;
  char file_name[FITS_MAX_CHAR];
  if(fits_file_name(fits_file_ptr, file_name, &status)){
    fits_report_error(stderr,status);
    throw std::runtime_error("Error reading fits file name");
  }
  return std::string(file_name);
}

int fits_get_img_ndim(fitsfile *fits_file_ptr){
  int status=0;
  int n_dim=0;
  if(fits_get_img_dim(fits_file_ptr, &n_dim, &status)){
    fits_report_error(stderr,status);
    throw std::runtime_error("Error reading hdu number of dimensions.");
  }
  return n_dim;
}

std::vector<long> fits_get_img_shape(fitsfile *fits_file_ptr){
  int status=0;
  int n_dim = 0;
  try{
    // set current hdu + read nb of dims
    n_dim = fits_get_img_ndim(fits_file_ptr);
  }catch(const std::exception& e){throw;}
  std::vector<long> shape(n_dim);
  if( fits_get_img_size(fits_file_ptr, n_dim, shape.data(), &status)){
    fits_report_error(stderr,status);
    throw std::runtime_error("Error reading hdu shape.");
  }
  return shape;
}


int fits_get_img_type(fitsfile *fits_file_ptr){
  int type;
  int status = 0;
  if(fits_get_img_type(fits_file_ptr,&type, &status)){
    fits_report_error(stderr,status);
    throw std::runtime_error("Error reading hdu type (bitpix).");
  }
  return type;
}

} // io
} // ships
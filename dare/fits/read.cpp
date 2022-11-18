#include <stdexcept>
#include <iostream>
#include <algorithm>

#include "read.hpp"
#include "interface.hpp"
#include "types.hpp"

namespace ships{
namespace io{
template<typename T>
T fits_get_key(fitsfile *fits_file_ptr, std::string key){
    int status=0;
    T value;
    if(fits_read_key(fits_file_ptr, fits_key_type_code<T>(), key.c_str(), &value, nullptr, &status)){
        fits_report_error(stderr,status);
        std::string err = "Error while reading the key '"+key+"'";
        throw std::runtime_error(err);
    }
    return value;
}
template int fits_get_key(fitsfile *fits_file_ptr, std::string key);
template long fits_get_key(fitsfile *fits_file_ptr, std::string key);
template float fits_get_key(fitsfile *fits_file_ptr, std::string key);
template double fits_get_key(fitsfile *fits_file_ptr, std::string key);

template<> std::string fits_get_key(fitsfile *fits_file_ptr, std::string key){
    int status=0;
    char value[FITS_MAX_CHAR];
    if(fits_read_keyword(fits_file_ptr, key.c_str(), value, nullptr, &status)){
        fits_report_error(stderr,status);
        std::string err = "Error while reading the key '"+key+"'";
        throw std::runtime_error(err);
    }
    return std::string(value);
};

std::string fits_get_current_hdu_name(fitsfile *fits_file_ptr){
    std::string hdu_name;
    // hdu indexed from 1
    if(fits_current_hdu(fits_file_ptr)==1){
        return "PRIMARY";
    }
    try{
        hdu_name=fits_get_key<std::string>(fits_file_ptr,"EXTNAME");
    }catch(const std::exception& e){
        std::cerr<<"HDU name not register as 'EXTNAME', trying 'HDUNAME'\n";
        hdu_name=fits_get_key<std::string>(fits_file_ptr,"HDUNAME");
    }
    //eliminate quotes and whitespaces
    std::remove(hdu_name.begin(),hdu_name.end(),' ');
    hdu_name.erase(std::remove(hdu_name.begin(),hdu_name.end(),'\''),
                   hdu_name.end());
    return hdu_name;
}

template<typename T>
void fits_get_data(fitsfile * fits_file_ptr, unsigned long n_Elem, T *data){
    int anynull;
    int nullval=0;
    int status = 0;
    long fpixel = 1;
    std::string err;
    int bitpix = fits_get_key<long>(fits_file_ptr,"BITPIX");

    if( fits_img_type_code<T>() != bitpix){
        err = "Inconsistent type : request '"  + get_type_name<T>() +
        "' while hdu holds " + get_type_of_fits_code(bitpix);
        throw std::runtime_error(err);
    }

    if( fits_read_img(fits_file_ptr, fits_key_type_code<T>(), fpixel, n_Elem,
         &nullval, data, &anynull, &status)){
            fits_report_error(stderr, status);
            err = "Error while reading hdu : "+
                fits_get_current_hdu_name(fits_file_ptr);
            throw std::runtime_error(err);
    }
}
template void fits_get_data(fitsfile * fits_file_ptr, unsigned long n_Elem,
    long *data);
template void fits_get_data(fitsfile * fits_file_ptr, unsigned long n_Elem,
    int *data);
template void fits_get_data(fitsfile * fits_file_ptr, unsigned long n_Elem,
    float *data);
template void fits_get_data(fitsfile * fits_file_ptr, unsigned long n_Elem,
    double *data);

} // io
} // ships
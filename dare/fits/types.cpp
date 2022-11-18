#include <map>
#include <typeindex>
#include <stdexcept>

#include "types.hpp"

namespace ships{
namespace io{
const std::map<std::type_index,int> FITS_IMG_TYPE_CODE= {
  {typeid((int)0)     , LONG_IMG},
  {typeid((long)0)    , LONGLONG_IMG},
  {typeid((float)0.)  , FLOAT_IMG},
  {typeid((double)0.) , DOUBLE_IMG}
};

const std::map<std::type_index,int> FITS_KEY_TYPE_CODE= {
  {typeid((int)0)           , TLONG},
  {typeid((long)0)          , TLONGLONG},
  {typeid((float)0.)        , TFLOAT},
  {typeid((double)0.)       , TDOUBLE},
  {typeid(std::string(""))  , TSTRING}

};


const std::map<std::type_index,std::string> DATA_TYPE_NAME= {
  {typeid((int)0)     , "int"},
  {typeid((long)0)    , "long"},
  {typeid((float)0.)  , "float"},
  {typeid((double)0.) , "double"}
};

template<typename T>
static int fits_img_type_code(){
    return FITS_IMG_TYPE_CODE.at(typeid((T)0));
}
template int fits_img_type_code<int>();
template int fits_img_type_code<long>();
template int fits_img_type_code<float>();
template int fits_img_type_code<double>();


template<typename T>
static int fits_key_type_code(){
    return FITS_KEY_TYPE_CODE.at(typeid((T)0));
}
template int fits_key_type_code<int>();
template int fits_key_type_code<long>();
template int fits_key_type_code<float>();
template int fits_key_type_code<double>();
template int fits_key_type_code<std::string>();

std::string get_type_of_fits_code(int fits_code){
    switch(fits_code){
        case LONG_IMG:
            return "32-bit   signed integers (int)";
        case LONGLONG_IMG:
            return "64-bit   signed integers (long)";
        case FLOAT_IMG:
            return "32-bit single precision floating point (float)";
        case DOUBLE_IMG:
            return "64-bit double precision floating point (double)";
        default:
            throw std::runtime_error("unknown fits type code");
    }
}

template<typename T>
std::string get_type_name(){
    return DATA_TYPE_NAME.at(typeid((T)0));
}
template std::string get_type_name<int>();
template std::string get_type_name<long>();
template std::string get_type_name<float>();
template std::string get_type_name<double>();

} // io
} // ships
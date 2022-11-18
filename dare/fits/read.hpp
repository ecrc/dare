#ifndef IO_FITS_READ_H
#define IO_FITS_READ_H
#include <string>
#include <vector>

#include <fitsio.h>

namespace ships{
namespace io{
template<typename T>
/*!
 * @brief Return the value associtaed to a keyword in the current fitsfile hdu
 *
 * @param fits_file_ptr : fitsfile* : fits file to read from
 * @param key : string : keyword's name
 */
T fits_get_key(fitsfile * fits_file_ptr, std::string key);

/*!
 * @brief Return the current hdu name
 *
 * @param fits_file_ptr : fitsfile* : fits file to read from
 */
std::string fits_get_current_hdu_name(fitsfile *fits_file_ptr);

template<typename T>
/*!
 * @brief Fill a pointer with the content of the current hdu
 *
 * @param fits_file_ptr : fitsfile* : fits file to read from
 * @param n_elem : unsigned long : number of element to read
 * @param data : T* : pointer to be filled with hdu data
 */
void fits_get_data(fitsfile *fits_file_ptr, unsigned long n_elem, T *data);

//template<typename T>
//void fits_get_data(fitsfile *fits_file_ptr, std::vector<T>  &data);

} // io
} // ships
#endif // IO_FITS_READ_H
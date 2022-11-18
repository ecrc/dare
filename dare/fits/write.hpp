#ifndef IO_FITS_WRITE_H
#define IO_FITS_WRITE_H
#include <string>
#include <vector>

#include <fitsio.h>

namespace ships{
namespace io{
//cannot use fits_write_key:
//cfitsio uses:
//#define fits_write_key          ffpky

/*!
 * @brief write a keyword with value to the current hdu
 *
 * @param fits_file_ptr : fitsfile* : fits file to write to
 * @param key : string : keyword's name
 * @param value : T : value associated to the keyword
 * @param comment : string : optional, comment on the keyword
 */
template<typename T>
void fits_set_key(fitsfile *fits_file_ptr, const std::string &key, T value,
  const std::string &comment);

/*!
 * @brief append an image hdu to a fitsfie
 *
 * @param fits_file_ptr : fitsfile* :  fits file to write to
 * @param dimensions : vector<long> : data dimensions
 * @param data : T* : pointer to the hdu data
 * @param hdu_key : string : name of the image
 */
template<typename T>
void fits_set_data(fitsfile *fits_file_ptr, std::vector<long> dimensions,
    const T *data, const std::string &hdu_key="");

} // io
} // ships
#endif // IO_FITS_WRITE_H
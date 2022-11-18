#ifndef IO_FITS_INTERFACE_H
#define IO_FITS_INTERFACE_H
#include <string>
#include <vector>

#include <fitsio.h>

namespace ships{
namespace io{
#define FITS_MAX_CHAR 80

/*!
 * @brief open a fits file and return a pointer to a fitsfile
 *
 * @param file_name : string : fits file name
 * @param mode : int : io mode
 */
fitsfile* fits_open(const std::string &file_name, int mode);
/*!
 * @brief close a fitsfile
 *
 * @param fits_file_ptr : fitsfile* : fitsfile to close
 * @param n_hdu : int : number of hdu
 */
void fits_close(fitsfile *fits_file_ptr, int n_hdu=1);
/*!
 * @brief Return the name of an opened fitsfile
 *
 * @param fits_file_ptr : fitsfile* : fits file
 */
std::string fits_file_name(fitsfile *fits_file_ptr);
/*!
 * @brief return the number of hdu contained in a fits file
 *
 * @param fits_file_ptr : fitsfile* : fits file
 */
int fits_nb_hdu(fitsfile *fits_file_ptr);
/*!
 * @brief Return the index of the current hdu in a fits file
 *
 * @param fits_file_ptr : fitsfile* : fits file
 */
int fits_current_hdu(fitsfile *fits_file_ptr);
/*!
 * @brief Set the current hdu in a fits file by index
 *
 * @param fits_file_ptr : fitsfile* : fits file
 * @param hdu_index : int : index of the current hdu
 */
void fits_set_hdu(fitsfile *fits_file_ptr, int hdu_index);
/*!
 * @brief Set the current hdu in a fits file by name
 *
 * @param fits_file_ptr : fitsfile* : fits file
 * @param hdu_name : string : name of the current hdu
 */
void fits_set_hdu(fitsfile *fits_file_ptr, const std::string &hdu_name);

/*!
 * @brief Return the number of dimensions of the fits file current hdu data
 *
 * @param fits_file_ptr : fitfile* : fits file
 */
int fits_get_img_ndim(fitsfile *fits_file_ptr);

/*!
 * @brief Return the shape of the fits file current hdu data
 *
 * @param fits_file_ptr : fitsfile* : fits file
 */
std::vector<long> fits_get_img_shape(fitsfile *fits_file_ptr);

/*!
 * @brief Return the type of the current hdu (as int)
 *
 * @param fits_file_ptr : fitsfile* : fits file
 */
int fits_get_img_type(fitsfile *fits_file_ptr);

} // io
} // ships
#endif // IO_FITS_INTERFACE_H
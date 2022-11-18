#ifndef IO_FITS_TYPES_H
#define IO_FITS_TYPES_H

#include <fitsio.h>
#include <string>


namespace ships{
namespace io{
/*!
 * @brief return the fits IMG code associated to a builtin type
 *
 * @tparam T : requested builtin type
 * @return int : fits img type code
 */
template<typename T>
int fits_img_type_code();

/*!
 * @brief return the fits KEY code associated to a builtin type
 *
 * @tparam T : requested builtin type
 * @return int : fits key type code
 */
template<typename T>
int fits_key_type_code();

/*!
 * @brief Return the description associated to a fits type code
 *
 * @param fits_code : int : fits type code
 */
std::string get_type_of_fits_code(int fits_code);

/*!
 * @brief Return the type name of a builtin type
 *
 * @tparam T : requested builtin type
 */
template<typename T>
std::string get_type_name();

} // io
} // ships
#endif // COMMON_IO_FITS_TYPES_H
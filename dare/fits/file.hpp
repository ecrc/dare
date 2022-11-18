#ifndef IO_FITS_FILE_H
#define IO_FITS_FILE_H
#include <string>
#include <vector>

#include <fitsio.h>

namespace ships{
namespace io{
class FitsFile{
  std::string m_file_name; //!< file name
  fitsfile *m_fits_ptr;    //!<
  int m_n_hdu;             //!< number of hdu in the fitsfile
  int m_mode;              //!< file io mode (read / write)

  public:
  /*! @brief Default constructor */
  FitsFile();
  /*!
   * @brief constructor
   *
   * @param file_name : string : fits file name
   * @param mode : string : io mode among
   *                              "r" : read
   *                              "w" : write
   *                              "a" : append
   *                              "w+": overwrite
   *
   */
  FitsFile(const std::string &file_name, const std::string &mode);
  /*! @brief copy constructor */
  FitsFile(const FitsFile &file);
  /*! @brief move constructor */
  FitsFile(FitsFile && file);
  ~FitsFile();

  void set(const std::string &file_name, const std::string &mode);

  FitsFile& operator=(const FitsFile& file);
  FitsFile& operator=(FitsFile &&file);
  bool operator==(const FitsFile &file);
  bool operator!=(const FitsFile &file);

  /*! @brief Return the name of all hdu */
  std::vector<std::string> hdu_names();

  /*! @brief move to an hdu
  *
  * @param hdu_index : int : index of the hdu to move to
  */
  void move_to_hdu(int hdu_index);
  /*! @brief move to an hdu
  *
  * @param hdu_name : string : name of the hdu to move to
  */
  void move_to_hdu(std::string hdu_name);

  /*! @brief Return the index of the current hdu*/
  int current_hdu_index();
  /*! @brief Return the name (string) of the current hdu*/
  std::string current_hdu_name();
  /*! @brief Return the index and name (string) of the current hdu*/
  std::string current_hdu();

  /*!
   * @brief Return the shape of a particular hdu's data
   *
   * @param hdu_index : int : hdu index
   */
  std::vector<long> get_data_shape(int hdu_index);
  /*!
   * @brief Return the shape of a particular hdu's data
   *
   * @param hdu_name : string : hdu name
   */
  std::vector<long> get_data_shape(std::string hdu_key);

  /*!
   * @brief Return the size (number of elements) of a particular hdu's data
   *
   * @param hdu_index : int hdu index
   */
  unsigned long get_data_size(int hdu_index);
  /*!
   * @brief Return the size (number of elements) of a particular hdu's data
   *
   * @param hdu_key : int hdu name
   */
  unsigned long get_data_size(std::string hdu_key);

  /*!
   * @brief Return the type of a given hdu (as int)
   *
   * @param hdu_index : int hdu index
   */
  int get_type(int hdu_index);

  /*!
   * @brief Return the type of a given hdu (as int)
   *
   * @param hdu_key : int hdu name
   */
  int get_type(const std::string &hdu_key);

  template<typename T>
  /*!
   * @brief Return the value associated to a keyword in a particular hdu
   *
   * @param hdu_index : int : hdu index
   * @param key : string : keyword's name
   */
  T get_key(int hdu_index, const std::string &key);
  template<typename T>
  /*!
   * @brief Return the value associated to a keyword in a particular hdu
   *
   * @param hdu_name : int : hdu name
   * @param key : string : keyword's name
   */
  T get_key(const std::string &hdu_key, const std::string &key);

  template<typename T>
  /*!
   * @brief Set a keyword value in a particular hdu
   *
   * @param hdu_index : int : hdu index
   * @param key : string : keyword's name
   * @param value : T : value associated to the keyword
   * @param comment : string : optional comment on the keyword
   */
  void set_key(int hdu_index, const std::string &key, T value,
    const std::string &comment="");
  template<typename T>
  /*!
   * @brief Set a keyword value in a particular hdu
   *
   * @param hdu_name : int : hdu name
   * @param key : string : keyword's name
   * @param value : T : value associated to the keyword
   * @param comment : string : optional comment on the keyword
   */
  void set_key(const std::string &hdu_key, const std::string &key,
    T value, const std::string &comment="");

  template<typename T>
  /*!
   * @brief Read data from a particular hdu
   *
   * @param hdu_index : int : hdu index
   * @param n_elem : unsigned long : number of elements to read
   * @param data : T* : pointer to read the data to
   */
  void get_data(int hdu_index, unsigned long n_elem, T *data);
  template<typename T>
  /*!
   * @brief Read data from a particular hdu
   *
   * @param hdu_name : string : hdu name
   * @param n_elem : unsigned long : number of elements to read
   * @param data : T* : pointer to read the data to
   */
  void get_data(const std::string &hdu_key, unsigned long n_elem, T *data);

  /*!
  * @brief Return a vector filled with a particular hdu's data
  *
  * @param hdu_index : int : hdu index
  * @return std::vector<T> : vector holding the hdu's data
  */
  template<typename T>
  std::vector<T> get_data(int hdu_index);
  /*!
  * @brief Return a vector filled with a particular hdu's data
  *
  * @param hdu_name : string : hdu name
  * @return std::vector<T> : vector holding the hdu's data
  */
  template<typename T>
  std::vector<T> get_data(const std::string &hdu_key);


  template<typename T>
  /*!
   * @brief Write an hdu image to a fits file
   *
   * @param shape : vector<long> : data's shape
   * @param data : T* : pointer to the data
   * @param hdu_key : string : hdu_name
   */
  void set_data(std::vector<long> shape, const T *data,
    const std::string &hdu_key="");

};

} // io
} // ships
#endif // IO_FITS_FILE_H
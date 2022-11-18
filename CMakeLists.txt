cmake_minimum_required(VERSION 3.18)
# set the project name
project(ddare C CXX CUDA)

set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 17)

############################################################
# choose Linear Algebra Runtime Backend! default chameleon
############################################################
OPTION(CHAMELEON_BACKEND "Use Chameleon as backend" OFF)
OPTION(DPLASMA_BACKEND "Use DPLASMA as backend" OFF)

# find deps
set(dep_lib "")
set(dep_inc "")

if (CHAMELEON_BACKEND AND DPLASMA_BACKEND)
    message(FATAL_ERROR "You can only use one backend!")
endif()

if (NOT CHAMELEON_BACKEND AND NOT DPLASMA_BACKEND)
    message(FATAL_ERROR "You must use one backend!")
endif()

if (CHAMELEON_BACKEND)
    message(STATUS "Using chameleon as backend!")
    set(BLA_VENDOR Intel10_64lp_seq)
    find_package(BLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CFITS REQUIRED IMPORTED_TARGET cfitsio)
    include(dare/fits/CMakeLists.txt)
    add_library(fitsio SHARED ${FITS_SRC})
    target_link_libraries(fitsio PUBLIC PkgConfig::CFITS)
    pkg_check_modules(STARPU REQUIRED IMPORTED_TARGET starpu-1.3)
    pkg_check_modules(CHAMELEON REQUIRED IMPORTED_TARGET chameleon)
    list(APPEND dep_lib PkgConfig::STARPU)
    list(APPEND dep_lib PkgConfig::CHAMELEON)
    list(APPEND dep_lib fitsio BLAS::BLAS)
    list(APPEND dep_inc ${STARPU_INCLUDE_DIRS})
    list(APPEND dep_inc ${CHAMELEON_INCLUDE_DIRS})
    set(APPSRC ddarechameleon)
    # app build
    foreach(appbin ${APPSRC})
        add_executable(${appbin} 
            dare/${appbin}.cpp
            dare/chameleonsrc/chameio.cpp
            dare/common/darecommon.cpp)
        target_include_directories(${appbin} PUBLIC ${dep_inc} dare)
        target_link_libraries(${appbin} PUBLIC ${dep_lib})
    endforeach()
endif()

if (DPLASMA_BACKEND)
    message(STATUS "Using DPLASMA as backend!")
    set(BLA_VENDOR Intel10_64lp_seq)
    find_package(BLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CFITS REQUIRED IMPORTED_TARGET cfitsio)
    include(dare/fits/CMakeLists.txt)
    add_library(fitsio SHARED ${FITS_SRC})
    target_link_libraries(fitsio PUBLIC PkgConfig::CFITS)
    pkg_check_modules(Eigen3 IMPORTED_TARGET eigen3)
    message(STATUS "EIGEN3 INC PATH" ${Eigen3_INCLUDE_DIRS})
    LIST(APPEND dep_inc ${Eigen3_INCLUDE_DIRS})
    pkg_check_modules(PARSEC REQUIRED IMPORTED_TARGET parsec)
    pkg_check_modules(DPLASMA REQUIRED IMPORTED_TARGET dplasma)
    find_package(MPI REQUIRED)
    list(APPEND dep_lib PkgConfig::PARSEC)
    list(APPEND dep_lib PkgConfig::DPLASMA)
    list(APPEND dep_lib ${MPI_C_LIBRARIES})
    list(APPEND dep_lib fitsio BLAS::BLAS)
    list(APPEND dep_inc ${PARSEC_INCLUDE_DIRS})
    list(APPEND dep_inc ${DPLASMA_INCLUDE_DIRS})
    list(APPEND dep_inc $ENV{MPI_ROOT}/include)
    set(APPSRC ddareparsec)
    # app build
    foreach(appbin ${APPSRC})
        add_executable(${appbin} 
            dare/${appbin}.cpp
            dare/common/darecommon.cpp 
            dare/parsecsrc/distdata.cpp
            dare/parsecsrc/util.cpp
            dare/parsecsrc/parsecio.cpp)
        target_include_directories(${appbin} PUBLIC ${dep_inc} dare)
        target_link_libraries(${appbin} PUBLIC ${dep_lib})
    endforeach()
endif()
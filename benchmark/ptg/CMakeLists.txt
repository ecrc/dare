cmake_minimum_required(VERSION 3.18)
project(dare_ptg C)

find_package(DPLASMA CONFIG REQUIRED)
include("${PARSEC_CMAKE_DIRS}/ParsecCompilePTG.cmake")

add_executable(ddareptg doubling.c)
target_compile_features(ddareptg PRIVATE c_std_11)
target_include_directories(ddareptg PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_ptg_sources(ddareptg PRIVATE add_identity.jdf)
target_link_libraries(ddareptg PRIVATE DPLASMA::dplasma LAPACKE::LAPACKE m)

cmake_minimum_required(VERSION 3.14)
#add_subdirectory(nasoq)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

    if(NOT DEFINED NASOQ_ROOT)
        set(NASOQ_ROOT_DIR "./nasoq")
    else()
        set(NASOQ_ROOT_DIR "${NASOQ_ROOT}")
    endif()

### LBL example
project(LBL_jl)
set(CMAKE_CXX_STANDARD 11)

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

# TODO: remove MKL dependency using OpenBLAS submodule
find_package(MKL REQUIRED)

set(SUITESPARSE_INCLUDE_DIR_HINTS ${SUITESPARSE_INCLUDE_DIR_HINTS} ${SUITE_ROOT_PATH}/include)
set(SUITESPARSE_LIBRARY_DIR_HINTS ${SUITESPARSE_LIBRARY_DIR_HINTS} ${SUITE_ROOT_PATH}/lib)
set(BLA_STATIC TRUE)
find_package(SuiteSparse OPTIONAL_COMPONENTS)

set( INCLUDE_INSTALL_DIR ${METIS_ROOT_PATH}/include)
set( LIB_INSTALL_DIR ${METIS_ROOT_PATH}/libmetis)
find_package(METIS REQUIRED)
set(METIS_LIBRARY "")
set(METIS_INCLUDE_DIR "")

include_directories(${METIS_INCLUDES} ${MKL_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}
        "${NASOQ_ROOT_DIR}/QP/" "${NASOQ_ROOT_DIR}/symbolic/"
        "${NASOQ_ROOT_DIR}/common/" "${NASOQ_ROOT_DIR}/ldl/"
        "${NASOQ_ROOT_DIR}/triangularSolve/" "${NASOQ_ROOT_DIR}/linear_solver/"
        "${NASOQ_ROOT_DIR}/gmres/"
        "${NASOQ_ROOT_DIR}/matrixVector/" "${NASOQ_ROOT_DIR}/matrixMatrix/")

set(SOURCE_FILES_NAS lbljl.cpp)
add_library(LBLJL SHARED ${SOURCE_FILES_NAS} lbljl.cpp)
target_link_libraries(LBLJL PRIVATE
        ${MKL_LIBRARIES} ${SUITESPARSE_LIBRARIES}
        ${METIS_LIBRARY} ${METIS_LIBRARIES} JlCxx::cxxwrap_julia)
if(OpenMP_CXX_FOUND)
    target_link_libraries(LBLJL PRIVATE OpenMP::OpenMP_CXX)
endif()
install (TARGETS LBLJL
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

########### already have **CasADi** and **IPOPT** ----> uncomment below.
# # Detect platform
# if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
#     set(CASADI_ROOT "${CMAKE_SOURCE_DIR}/third_party/casadi/linux")
#     set(IPOPT_ROOT  "${CMAKE_SOURCE_DIR}/third_party/ipopt/linux")
# else()
#     message(FATAL_ERROR "This project currently supports Linux only.")
# endif()

# # CasADi
# add_library(casadi SHARED IMPORTED)
# set_target_properties(casadi PROPERTIES
#     IMPORTED_LOCATION "${CASADI_ROOT}/libcasadi.so"
#     INTERFACE_INCLUDE_DIRECTORIES "${CASADI_ROOT}"
# )

# # Ipopt
# add_library(ipopt SHARED IMPORTED)
# set_target_properties(ipopt PROPERTIES
#     IMPORTED_LOCATION "${IPOPT_ROOT}/libipopt.so"
#     INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_ROOT}/include"
# )

# # Link them to your executable
# target_link_libraries(your_target PRIVATE casadi ipopt)

########### If using the bundled **CasADi** and **IPOPT** ----> uncomment below.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(CtrModel)

# general compilation flags
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /std:c++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")
endif()

# Boost library (required by gnuplot-iostream, used for plotting)
FIND_PACKAGE( Boost 1.40 COMPONENTS iostreams system filesystem REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
LINK_LIBRARIES( ${Boost_LIBRARIES} )

# OpenMP library (used for parallel computing)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM Library)
include_directories(SYSTEM Library/eigen-3.4.0)
include_directories(SYSTEM src)

add_subdirectory(src)
add_subdirectory(ctcr_casadi_wrappers)
add_subdirectory(demo)

set(CMAKE_DISABLE_SOURCE_CHANGES OFF)
set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)


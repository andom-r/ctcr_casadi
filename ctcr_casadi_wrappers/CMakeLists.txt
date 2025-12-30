
########### already have **CasADi** and **IPOPT** ----> uncomment below.
# # Find CasADi
# find_path(CASADI_INCLUDE_DIR casadi/casadi.hpp
#   PATHS /usr/local/include
# )

# find_library(CASADI_LIBRARY casadi
#   PATHS /usr/local/lib
# )

# if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
#   message(FATAL_ERROR "CasADi not found")
# endif()

# # ---- CasADi wrapper library ----
# add_library(ctcrCasADiWrappers STATIC
#   BlackboxConstrYu0.cpp
#   # add more .cpp files here
# )

# target_include_directories(ctcrCasADiWrappers PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}       # wrapper headers
#   ${CASADI_INCLUDE_DIR}             # CasADi headers
#   ${CMAKE_SOURCE_DIR}/src           # CTCR core headers
# )

# target_link_libraries(ctcrCasADiWrappers PUBLIC
#   ctrModelLib        # your CTCR core library
#   ${CASADI_LIBRARY}
# )

# # Optimization flags
# if(MSVC)
#   target_compile_options(ctcrCasADiWrappers PRIVATE /O2 /DEIGEN_NO_DEBUG)
# else()
#   target_compile_options(ctcrCasADiWrappers PRIVATE -O3 -DEIGEN_NO_DEBUG)
# endif()


########### If using the bundled **CasADi** and **IPOPT** ----> uncomment below.
# ---- Paths to bundled libraries ----
set(CASADI_ROOT "${CMAKE_SOURCE_DIR}/third_party/casadi/linux")
set(IPOPT_ROOT  "${CMAKE_SOURCE_DIR}/third_party/ipopt/linux")

# ---- CasADi imported library ----
add_library(casadi SHARED IMPORTED)
set_target_properties(casadi PROPERTIES
    IMPORTED_LOCATION "${CASADI_ROOT}/libcasadi.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CASADI_ROOT}"
)

# ---- CasADi Ipopt plugin (optional but recommended) ----
add_library(casadi_ipopt SHARED IMPORTED)
set_target_properties(casadi_ipopt PROPERTIES
    IMPORTED_LOCATION "${CASADI_ROOT}/libcasadi_nlpsol_ipopt.so"
)

# ---- Ipopt imported library ----
add_library(ipopt SHARED IMPORTED)
set_target_properties(ipopt PROPERTIES
    IMPORTED_LOCATION "${IPOPT_ROOT}/libipopt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_ROOT}/include"
)

# ---- Wrapper library ----
add_library(ctcrCasADiWrappers STATIC
    BlackboxConstr.cpp
    BlackboxConstrJacobian.cpp
    BlackboxConstrYu0.cpp
)

target_include_directories(ctcrCasADiWrappers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # wrapper headers
    ${CASADI_ROOT}                # CasADi headers
    ${CMAKE_SOURCE_DIR}/src       # CTCR core headers
)

target_link_libraries(ctcrCasADiWrappers PUBLIC
    ctrModelLib
    casadi
    casadi_ipopt
    ipopt
)

# ---- Optimization flags ----
if(MSVC)
    target_compile_options(ctcrCasADiWrappers PRIVATE /O2 /DEIGEN_NO_DEBUG)
else()
    target_compile_options(ctcrCasADiWrappers PRIVATE -O3 -DEIGEN_NO_DEBUG)
endif()

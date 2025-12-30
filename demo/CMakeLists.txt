########### already have **CasADi** and **IPOPT** ----> uncomment below.
# # Find CasADi
# find_path(CASADI_INCLUDE_DIR casadi/casadi.hpp
#           PATHS /usr/local/include
#           NO_DEFAULT_PATH)

# find_library(CASADI_LIBRARY NAMES casadi
#              PATHS /usr/local/lib
#              NO_DEFAULT_PATH)

# if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
#     message(FATAL_ERROR "CasADi not found. Please install it or adjust the paths.")
# endif()


# file(GLOB SRC_DEMO "*.cpp") # get all demo source files
# foreach (FILE ${SRC_DEMO}) 
#     get_filename_component(NAME_WITH_EXT ${FILE} NAME)
#     string(REPLACE ".cpp" "" TARGET_NAME "${NAME_WITH_EXT}") # set demo application name as the filename without the extension

#     add_executable (${TARGET_NAME} ${FILE})
    
#     # link the ctrModelLib
#     target_link_libraries(${TARGET_NAME} PUBLIC ctrModelLib) 

#     # Include CasADi
#     target_include_directories(${TARGET_NAME} PUBLIC ${CASADI_INCLUDE_DIR})
#     target_link_libraries(${TARGET_NAME} PUBLIC ${CASADI_LIBRARY})

#     # link the ctcrCasADiWrappers
#     target_link_libraries(${TARGET_NAME} PUBLIC ctcrCasADiWrappers) 

#     if(MSVC) # enable optimization
#       target_compile_options(${TARGET_NAME} PRIVATE /O2 /DEIGEN_NO_DEBUG)
#     else()
#       target_compile_options(${TARGET_NAME} PRIVATE -O3 -DEIGEN_NO_DEBUG)
#     endif()
# endforeach ()



########### If using the bundled **CasADi** and **IPOPT** ----> uncomment below.
# ---- Use bundled CasADi and Ipopt ----
set(CASADI_ROOT "${CMAKE_SOURCE_DIR}/third_party/casadi/linux")
set(IPOPT_ROOT  "${CMAKE_SOURCE_DIR}/third_party/ipopt/linux")

# CasADi main library
add_library(casadi SHARED IMPORTED)
set_target_properties(casadi PROPERTIES
    IMPORTED_LOCATION "${CASADI_ROOT}/libcasadi.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CASADI_ROOT}"
)

# CasADi Ipopt plugin
add_library(casadi_ipopt SHARED IMPORTED)
set_target_properties(casadi_ipopt PROPERTIES
    IMPORTED_LOCATION "${CASADI_ROOT}/libcasadi_nlpsol_ipopt.so"
)

# Ipopt library
add_library(ipopt SHARED IMPORTED)
set_target_properties(ipopt PROPERTIES
    IMPORTED_LOCATION "${IPOPT_ROOT}/libipopt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_ROOT}/include"
)

file(GLOB SRC_DEMO "*.cpp")

foreach (FILE ${SRC_DEMO})
    get_filename_component(NAME_WITH_EXT ${FILE} NAME)
    string(REPLACE ".cpp" "" TARGET_NAME "${NAME_WITH_EXT}")

    add_executable(${TARGET_NAME} ${FILE})

    # Link your core model
    target_link_libraries(${TARGET_NAME} PUBLIC ctrModelLib)

    # Link your CasADi wrapper library
    target_link_libraries(${TARGET_NAME} PUBLIC ctcrCasADiWrappers)

    # Link bundled CasADi + Ipopt
    target_link_libraries(${TARGET_NAME} PUBLIC casadi casadi_ipopt ipopt)

    # Include CasADi headers
    target_include_directories(${TARGET_NAME} PUBLIC ${CASADI_ROOT})

    # Optimization flags
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /O2 /DEIGEN_NO_DEBUG)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -O3 -DEIGEN_NO_DEBUG)
    endif()
endforeach()
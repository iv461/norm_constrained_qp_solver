cmake_minimum_required(VERSION 3.15)
project(norm_constrained_qp_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("-fPIC") 
    add_compile_options("-Wno-deprecated")
    add_compile_options("-Werror=return-type")
    add_compile_options("-O3") # Enable vectorization using O3
    add_compile_options("-Werror=return-local-addr")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") 
    #add_compile_options("/arch:AVX /bigobj")
    add_compile_options("/bigobj")
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)

pybind11_add_module(norm_constrained_qp_solver
    src/python_bindings.cpp
)

target_link_libraries(norm_constrained_qp_solver PUBLIC
    Eigen3::Eigen
    pybind11::pybind11
    fmt::fmt
    )
    
# Below lines for installing the pybind module as a python package:
configure_file(src/norm_constrained_qp_solver/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/norm_constrained_qp_solver/__init__.py)
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
install(TARGETS norm_constrained_qp_solver DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/norm_constrained_qp_solver")
install(CODE "execute_process(COMMAND pip install .)")


# Add the test directory
enable_testing()
add_subdirectory(test)
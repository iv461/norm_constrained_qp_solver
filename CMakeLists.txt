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
    add_compile_options("/bigobj")
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt REQUIRED)
find_package(spectra REQUIRED)

add_library(norm_constrained_qp_solver INTERFACE)

target_include_directories(
    norm_constrained_qp_solver
    INTERFACE include/
)

target_link_libraries(norm_constrained_qp_solver INTERFACE
    Eigen3::Eigen
    Spectra::Spectra
    fmt::fmt
    )
 
# Add the test directory
enable_testing()
add_subdirectory(test)
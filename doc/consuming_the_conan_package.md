# Consuming the conan package

To use the solver after installing the conan pacckage, create a C++-project 
with that constains the following lines in the `CMakeLists.txt`-file

```Cmake
[...]
find_package(norm_constrained_qp_solver REQUIRED)

[...]

add_executable(my_executable ...)

target_link_libraries(my_executable PUBLIC
    norm_constrained_qp_solver::norm_constrained_qp_solver
    [...]
    )
[...]
```
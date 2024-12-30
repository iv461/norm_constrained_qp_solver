# Find GoogleTest package
find_package(GTest REQUIRED)

# Add the test executable
add_executable(tests test_solver.cpp)

# Link the test executable with GTest and your library
target_link_libraries(tests PRIVATE 
    gtest::gtest 
    Spectra::Spectra
    Eigen3::Eigen
    fmt::fmt
    GTest::Main
    )

# Add tests to the CTest suite
add_test(NAME NCSSolverTests COMMAND tests)
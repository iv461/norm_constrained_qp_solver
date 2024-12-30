# Solver for norm-constrained optimization problems 

# Introduction

This is a tiny and fast solver for solving the non-convex optimization problem with unit-norm equality constraint to global optimality:

```math
\begin{equation}
	\begin{aligned}
		\min_{x \in \mathbb{R}^d} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{C} \mathbf{x} - \mathbf{b}^T \mathbf{x}\\
		\text{subject to} \quad  & \lVert \mathbf{x} \rVert = s\\
	\end{aligned}
\end{equation}
```

where $\mathbf{A}$ is a symmetric matrix (does not need to be positive definite) and $s$ is a parameter. This problem is also known as the Trust-Region Subproblem (TRS).

The solver is optimized for small, fixed-sized matrices.
It is based on eigen-decomposition with subsequent root-finding.

# Dependencies 

- C++17
- Eigen linear algebra library, version >= `3.4`

# Install 

Install using the conan package manager: 

```sh
conan install norm_constrained_qp_solver
```

Add this to your `CMakeLists.txt`-file:

```Cmake
find_package(norm_constrained_qp_solver REQUIRED)
add_executable(my_executable ...)

target_link_libraries(my_executable PUBLIC
    norm_constrained_qp_solver::norm_constrained_qp_solver
    [...]
    )
```

# Example usage 

```c++
#include <norm_constrained_qp_solver.hpp>

Eigen::Matrix3<double> C = Eigen::Matrix3<double>::Ones();
Eigen::Vector3<double> b = Eigen::Vector3<double>::Random();
double s = 1.;

Eigen::Vector3<double> optimal_x = ncs::solve_norm_constrained_qp(C, b, s);
```

# Build from source 

To build from source and run the tests, see [Building from source](doc/build_from_source.md)

# License 

This code is licensed under the MIT license.

# Related projects 

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) Julia package, optimized for large and sparse matrices, with  additional linear constraints
- [QPnorm.jl](https://github.com/oxfordcontrol/QPnorm.jl) Julia package, supports additionally a minimum norm constraint 
- [Manopt](https://www.manopt.org/) Matlab/Python/Julia package for manifold optimization, supports optimization over the sphere (=norm-constraint), but without global optimality guarantee

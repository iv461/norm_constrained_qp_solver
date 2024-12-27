# Solver for norm-constrained optimization problems 

# Introduction

This is a tiny and fast solver for solving the non-convex optimization problem with unit-norm equality constraint to global optimality:

```math

\begin{equation}
	\begin{aligned}
		\min_{x \in \mathbb{R}^d} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} - \mathbf{g}^T \mathbf{x}\\
		\text{subject to} \quad  & \lVert \mathbf{x} \rVert \leq s\\
	\end{aligned}
\end{equation}
```

where $\mathbf{A}$ is a symmetric matrix (does not need to be positive definite) and $s$ is a parameter.

This problem is also known as the Trust-Region Subproblem (TRS).
The solver is based on eigen-decomposition with subsequent root-finding.

The solver is also fast for small fixed-sized matrices.

# Dependencies 

- C++17
- Eigen linear algebra libracy, version >= 3.4

# Install 

Install the conan package: 
TODO 


# Build from source 

[Build from source](doc/build_from_source.md)

# Use trough python

The solver can be called from Python: 


# License 

This code is lincensed under the MIT licence.

# Related projects 

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) Julia package, offers additional features such as ellipsoidal norms and additional linear constraints
- [QPnorm.jl](https://github.com/oxfordcontrol/QPnorm.jl) Julia package, supports additionally a minimum norm constraint 
- [Manopt](https://www.manopt.org/) Matlab/Python/Julia package, able to solve the problem over the sphere (= norm-constraint), but without global optimality guarantee

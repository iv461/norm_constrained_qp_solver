# Solver for norm-constrained optimization problems 

# Introduction

This is a tiny and fast solver for solving the non-convex optimization problem with unit-norm equality constraint to global optimality:

```math

\begin{equation}
	\begin{aligned}
		\min_{x \in \mathbb{R}^d} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{g}^T \mathbf{x}\\
		\text{subject to} \quad  &|| \mathbf{x} || \leq 1\\
	\end{aligned}
\end{equation}
```

where $\mathbf{A}$ is a symmetric matrix (does not need to be positive definite).

This problem is also known as the Trust-Region Subproblem (TRS).
The norm-constraint does not have to be 1 -- it can be any scale s. For this, simply scale the data-matrices $\mathbf{A}$ and $\mathbf{g}$ by $\frac{1}{s}$.

The solver is based on eigen-decomposition with subsequent root-finding.

The solver is fast for small fixed-sized matrices, but is equally capable of solving for large sparse matrices by using the SPECTRA library.

# Dependencies 
- C++ 17
- Eigen linear algebra libracy, version >= 3.4

# Install 

Install the conan package: 

# Use trough python

The solver can be called from Python: 



# License 

This code is lincensed under the MIT licence.

# Related projects 

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) Julia package, offers additional features such as ellipsoidal norms and additional linear constraints


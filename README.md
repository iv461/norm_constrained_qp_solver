# Solver for norm-constrained optimization problems 

# Introduction

This is a tiny and fast solver for solving the non-convex optimization problem with unit-norm equality constraint to global optimality:

```math

\begin{equation}
	\begin{aligned}
		\min_{x \in \mathbb{R}^d} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{g}^T \mathbf{x}\\
		\text{subject to} \quad  & \lVert \mathbf{x} \rVert \leq 1\\
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
TODO 


# Build from source 

If you want to build the project from source instead of installing it, do the following: 

Tested on:

- Windows
- Ubuntu Linux 22.04 LTS

Thanks to conan, this package is not dependent on the system package manager on Linux, therefore it probabably works on other Linux distros too, just give it a try !

Windows: Do not use the MS-Store Python-version, as this may lead to the conan-command not being installed correctly, instead install Python from python.org


Install conan2: 
```sh
pip install conan~=2.0
```

Do this once:

```sh
conan profile detect
```

Now install the dependencies with conan:

```sh
conan install . --build=missing --output build
```

Compile and install the pip package with CMake: 

### Linux 

```sh
mkdir Release
cd Release
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
```

### Windows
- Install newest Visual Studion Compiler.

```sh
mkdir Release
cd Release
cmake .. -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE="conan_toolchain.cmake"
cmake --build . --config Release --target install
```


# Use trough python

The solver can be called from Python: 



# License 

This code is lincensed under the MIT licence.

# Related projects 

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) Julia package, offers additional features such as ellipsoidal norms and additional linear constraints
- [QPnorm.jl](https://github.com/oxfordcontrol/QPnorm.jl) Julia package, supports additionally a minimum norm constraint 

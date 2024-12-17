/// TODO license 


#include <pybind11/eigen.h>
#include <pybind11/functional.h
#include <pybind11/pybind11.h>

#include "norm_constrained_qp_solver.hpp"

namespace py = pybind11;
PYBIND11_MODULE(norm_constrained_qp_solver, m) {
     m.def("solve_norm_constrained_qp", &solve_norm_constrained_qp);
}
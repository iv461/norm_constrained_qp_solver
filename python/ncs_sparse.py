import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh

def solve_ncs_sparse(C, b, s):
    dims = C.shape[0]
    M = np.zeros((2*dims, 2*dims))

    M[:dims, :dims] = -C
    M[dims:, dims:] = -C
    M[dims:, :dims] = np.eye(dims)
    M[:dims, dims:] = np.outer(b, b) / (s**2)
    #print(f"M: {M}")

    # Find the largest eigenvector using ARPACK
    eigenvalues, eigenvectors = eigs(M, 1, which = 'LR')
    z_star = eigenvectors.real

    print(f"Best eigenvalue: {eigenvalues.real}, z_star: {z_star}")
    z1_star = z_star[:dims]
    z2_star = z_star[dims:]
    print(f"z1_star: {z1_star}, z2_star: {z2_star}")
    #print(f"R: {res}")

    z1_star_norm = np.linalg.norm(z1_star)
    if z1_star_norm  < 1e-9: 
        print("HARD CASE")
        return None
    
    dp = np.dot(b, z2_star)
    sign_dp = dp / abs(dp)
    x_opt = - sign_dp * s * (z1_star / z1_star_norm)
    return x_opt
    
def test1(): 
    C = np.array([[ 1.59616715,  1.57653008,  0.90479527],
                  [ 1.57653008,  0.75925698,  1.57438616],
                  [ 0.90479527,  1.57438616, -2.55542413]])
    b = -np.array([ 1.95181977, -0.87145519, -0.95380709])
    s = 1. 

    x_hat1 = solve_ncs_sparse(C, b, s)
    best_sol = np.array([0.4996741, -0.06190374, -0.86399868])
    diff = np.linalg.norm(best_sol - x_hat1.squeeze())
    assert diff < 1e-3, f"Diff is: {diff}"

def test2(): 

      
    C = np.array([[ -1.07822383, -2.78673686, -1.23438251],
                  [ -2.78673686, 0.93347297, 0.54945616],
                  [ -1.23438251, 0.54945616, -0.05524914]])
    
    b = -np.array([-0.68618036, -0.29540059, -0.51183855])
    s = 1. 

    x_hat1 = solve_ncs_sparse(C, b, s)
    print(f"x_hat1: {x_hat1}")
    
    best_sol = np.array([-0.81721938, -0.48254904, -0.3151173])

    diff = np.linalg.norm(best_sol - x_hat1.squeeze())
    assert diff < 1e-3, f"Diff is: {diff}"

test1()
test2()
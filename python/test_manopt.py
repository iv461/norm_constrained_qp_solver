import numpy as np
import pymanopt

def solve_norm_constrained_qp(C, b):
    dims = C.shape[0]    
    manifold = pymanopt.manifolds.Sphere(dims)

    @pymanopt.function.autograd(manifold)
    def cost(x):
        return 0.5 * x.T @ C @ x - np.dot(b, x)

    problem = pymanopt.Problem(manifold, cost)
    solver = pymanopt.optimizers.TrustRegions(verbosity=1)
    result = solver.run(problem)
    return result.point, result.cost


def test_hard_case():
    dims = 3
    D = np.random.normal(size=(dims, dims))

    Q, _ = np.linalg.qr(D) # That's not how one uniformly samples the Stiefel manifold, but whatever
    eigenvals = np.array([-3.2, -0.3, 3.3])
    
    C = Q @ np.diag(eigenvals) @ Q.T

    #random_axis = np.array([.234976, .58736, -0.654])
    #b = np.cross(Q[:, 0], random_axis) # Make b orthogonal to one of the eigenvectors so that one of the d_i is zero.
    b = np.random.normal(size=(3,))
    # C = np.array([[-1.07822383, -2.78673686, -1.23438251],
    #                 [-2.78673686, 0.93347297, 0.54945616],
    #                 [-1.23438251, 0.54945616, -0.05524914]])
    # b = np.array([-0.68618036, -0.29540059, -0.51183855])
    
    # C = np.array([[ 0.16904076, -0.57421902,  1.08854251],
    #               [-0.57421902, -1.21054522, -2.82741677],
    #               [ 1.08854251, -2.82741677,  0.84150445]])
    
    # b = np.array([-0.8726189, 0.10428207, -0.21986756])

     
    print(f"Problem instance: C: {C}, b: {b}")
    
    min_cost = None
    best_sol = None
    for i in range(100):
        x_hat, cost = solve_norm_constrained_qp(C, b)
        if min_cost is None or min_cost > cost:
            print(f"Found better cost: {cost}, previous was: {min_cost}")
            found_both = min_cost is not None
            min_cost = cost
            best_sol = x_hat
            if found_both: break


    print(f"Best solution: {best_sol}, cost: {min_cost}")

    #print(f"Solution: {solve_norm_constrained_qp(C, b)}")

def test_hard_case2():
    dims = 2
    D = np.random.normal(size=(dims, dims))

    Q, _ = np.linalg.qr(D) # That's not how one uniformly samples the Stiefel manifold, but whatever
    eigenvals = np.array([-3.2, 3.3])
    
    C = Q @ np.diag(eigenvals) @ Q.T
    b = Q[:,0] * 0.567

    #d = Q.T @ b
    #print(f"d: {d}")

    C = np.array([[ 3.29193704,  0.22878861],
                    [0.22878861, -3.19193704]])    
    b = np.array([-0.0199698, 0.56664822])

    print(f"Problem instance: C: {C}, b: {b}")
    # Now search until we found both stationary points
    min_cost = None
    best_sol = None
    for i in range(100):
        x_hat, cost = solve_norm_constrained_qp(C, b)
        if min_cost is None or min_cost > cost:
            print(f"Found better cost: {cost}, previous was: {min_cost}")
            found_both = min_cost is not None
            min_cost = cost
            best_sol = x_hat
            if found_both: break


    print(f"Best solution: {best_sol}, cost: {min_cost}")


def test_algo2():
    C = np.array([[-1.07822383, -2.78673686, -1.23438251],
                    [-2.78673686, 0.93347297, 0.54945616],
                    [-1.23438251, 0.54945616, -0.05524914]])
    b = np.array([-0.68618036, -0.29540059, -0.51183855])

    s = 1. 

    dims = C.shape[0]
    M = np.zeros((2*dims, 2*dims))

    M[:dims, :dims] = -C
    M[dims:, dims:] = -C
    M[dims:, :dims] = np.eye(dims)
    M[:dims, dims:] = np.outer(b, b) / (s**2)
    print(f"M: {M}")

    res = np.linalg.eig(M)
    print(f"R: {r}")


    
def test_C_zero():
    dims = 3
    
    C = np.zeros((dims, dims))
    b = np.random.normal(size=(dims,))

    
    print(f"Problem instance: C: {C}, b: {b}")
    # Now search until we found both stationary points
    min_cost = None
    best_sol = None
    for i in range(100):
        x_hat, cost = solve_norm_constrained_qp(C, b)
        if min_cost is None or min_cost > cost:
            print(f"Found better cost: {cost}, previous was: {min_cost}")
            min_cost = cost
            best_sol = x_hat

    print(f"Best solution: {best_sol}, cost: {min_cost}")

    dot_prod = best_sol.dot(b) / (np.linalg.norm(b) * np.linalg.norm(best_sol))
    print(f"dot_prod: {dot_prod}")

test_hard_case()
#test_algo2()
#test_hard_case()
#test_hard_case2()
#test_C_zero()
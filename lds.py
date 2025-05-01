import numpy as np

from ssa import tangent_retract, sa, ssa


def lds(A: np.array, x_0: np.array, dt: float = 0.01, beta: float=0.1, iters=20):

    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(x_0.shape) == 1
    assert A.shape[0] == x_0.shape[0]

    n = A.shape[0]
    
    xs = np.zeros(shape=(iters, n))    
    xs[0, :] = x_0

    for iter_idx in range(1,iters):
        # xs[iter_idx] = xs[iter_idx - 1] + dt * (A @ xs[iter_idx - 1] - beta * xs[iter_idx - 1])
        xs[iter_idx] = xs[iter_idx - 1] + dt * (A @ xs[iter_idx - 1])
        
    return xs


def linear_hebbian_stable(A: np.array, x_0: np.array, eps: float, target: float, dt: float = 0.01, beta: float=0.1, dA: float = 0.1, iters=20):

    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert len(x_0.shape) == 1
    assert A.shape[0] == x_0.shape[0]

    n = A.shape[0]
    
    xs = np.zeros(shape=(iters, n))    
    xs[0, :] = x_0

    tangent_thetas = np.zeros(shape=(iters,))
    abscs = np.zeros(shape=(iters,))
    s_abscs = np.zeros(shape=(iters,))    
    s_norms = np.zeros(shape=(iters,))
    F_norms = np.zeros(shape=(iters,))

    for iter_idx in range(1,iters):
        xs[iter_idx] = xs[iter_idx - 1] + dt * (A @ xs[iter_idx - 1] - beta * xs[iter_idx])

        # hebbian correlation
        V = np.outer(xs[iter_idx], xs[iter_idx])

        # move in this direction, projected to the manifold
        A, theta = tangent_retract(A, V, eps, target, dA * dt) # dA and dt relationship... scaling by manifold size...

        # record angle to the tangent plane        
        tangent_thetas[iter_idx] = theta

        # find abscissa
        abscs[iter_idx] = sa(A)
        s_abscs[iter_idx] = ssa(A, eps, grad=False, start_at_absc=True)[0]
        s_norms[iter_idx] = np.linalg.norm(A, ord=2)
        F_norms[iter_idx] = np.linalg.norm(A, ord="fro")

        
        
    return xs, tangent_thetas, abscs, s_abscs, s_norms, F_norms

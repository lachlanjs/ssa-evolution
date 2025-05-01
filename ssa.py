"""
Functions for dealing with the smoothed spectral abscissa detailed in:
Vanbiervliet, Joris, et al. "The smoothed spectral abscissa for robust stability optimization." SIAM Journal on Optimization 20.1 (2009): 156-171.
https://epubs.siam.org/doi/abs/10.1137/070704034

"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, null_space
from scipy.optimize import root_scalar

from tqdm import tqdm

def sa(A: np.array):    

    return np.max(np.array([np.real(x) for x in np.linalg.eigvals(A)]))

def fAs(A: np.array, s: float, grad=True, ret_QP=True):
    """ find $f(A, s)$ where $s > \alpha(A)$ (not checked)

    Args:
        A (np.array): square matrix
        s (float): real value, equal to ssa(A, eps) when fAs(A, s) = 1.0/eps

    Returns:
        tuple: (f(A, s), par f(A, s) / par s, par f(A, s) / par A), singleton if grad=False
    """

    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    
    # solve Lyapunov:
    P = solve_continuous_lyapunov(A - s * np.identity(n), -1 * np.identity(n))
    Q = solve_continuous_lyapunov(A.T - s * np.identity(n), -1 * np.identity(n))    

    if grad:
        QP = Q@P
        # (f(A, s), par f(A, s) / par s, par f(A, s) / par A)
        if ret_QP:
            return (np.trace(P), -2.0 * np.trace(QP), 2.0 * QP, Q, P) # AHHHHH
        else:
            return (np.trace(P), -2.0 * np.trace(QP), 2.0 * QP)
    else:
        # (f(A, s),)        
        return (np.trace(P),)

def ssa(A: np.array, eps: float, maxiter: int = 20, mach_eps: float = 1e-5, grad=True, ret_QP=False, start_at_absc=True, method="newton"):
    """calculate the smoothed spectral abscissa $\alpha_{\epsilon}(A)$

    Args:
        A (np.array): square matrix
        eps (float): smoothing parameter value (>0)
    
        maxiter (int, optional): _description_. Defaults to 20.
        mach_eps (float, optional): _description_. Defaults to 1e-5.
        grad (bool, optional): _description_. Defaults to True.
        start_at_abscissa (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: (s, del_A) - only s if grad=False
    """
        
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    assert eps > 0

    # NOTE: apparently the reciprocal of f is better behaved numerically
    def recp_fAs_minus_eps(x: float, grad_recp=True):
        if grad_recp:
            y, ydels = fAs(A, x)[0:2]
            return (1.0/y) - eps, -1.0 * ydels / np.power(y, 2)
        else:
            y = fAs(A, x, grad=False)[0]
            return (1.0/y) - eps

    # root-find to solve the equation (1.0 / f(A, s)) - eps = 0 
    # for s
    #     
    # NOTE: not sure about bracket yet
    if start_at_absc:
        s_absc = sa(A)
        if method == "newton":
            
            s = root_scalar(lambda x: recp_fAs_minus_eps(x), method="newton", x0=s_absc + eps/10.0, maxiter=maxiter, fprime=True)
        elif method == "brentq":
            s = root_scalar(lambda x: recp_fAs_minus_eps(x, grad_recp=False), method="brentq", bracket=[s_absc + eps/10.0, s_absc + eps * 10.0], maxiter=maxiter) 
        else:
            raise Exception("Method must be one of [\"newton\", \"brentq\"]")
    else:
        if method == "newton":
            s = root_scalar(lambda x: recp_fAs_minus_eps(x), method="newton", x0=5.0, maxiter=maxiter, fprime=True)
        elif method == "brentq":            
            s = root_scalar(lambda x: recp_fAs_minus_eps(x, grad_recp=False), method="brentq", bracket=[-5.0, 5.0], maxiter=maxiter)    
        else:
            raise Exception("Method must be one of [\"newton\", \"brentq\"]")
        

    s, converged = s.root, s.converged
    if not converged:
        print("DID NOT CONVERGE")

    if grad:
        _, del_s, del_A, Q, P = fAs(A, s)
        if ret_QP:
            return s, -2.0 * del_A / del_s, Q, P # NOTE: was missing a 2.0 heres, NOTE: then I forgot the trace term as well
        else:
            return s, -2.0 * del_A / del_s # NOTE: was missing a 2.0 heres, NOTE: then I forgot the trace term as well

    return (s,)

def optimize_ssa(A: np.array, eps: float, target: float, step_size: float=0.0002, iters: int = 20, show_prog: bool=False):
    """ Optimizes the ssa using graient steps

    Args:
        A (np.array): _description_
        eps (float): _description_
        target (float): _description_
        step_size (float): _description_
        iters (int, optional): _description_. Defaults to 20.
    """

    s_abscs = []
    ss_abscs = []

    ran = tqdm(range(iters)) if show_prog else range(iters)
    for iter in ran:

        # get spectral abscissa for the record
        s_abscs.append(sa(A))

        # get smoothed spectral abscissa and gradient
        ss_absc, ss_absc_grad = ssa(A, eps)
        ss_abscs.append(ss_absc)

        # take step
        A += step_size * (target - ss_absc) * ss_absc_grad        


    return A, s_abscs, ss_abscs

def ssa_newton_retraction(A: np.array, V: np.array, eps: float, target: float):

    # V should be in the null-space of the differential of the ssa (tangent)
    A_new = A + V

    # compute ssa and grad
    ss_absc, ss_absc_grad = ssa(A_new, eps)
        
    # take step back to the target
    A_new += (target - ss_absc) * ss_absc_grad        

    return A_new

def tangent_retract(A: np.array, V: np.array, eps: float, target: float, step_size: float=0.01):

    n = A.shape[0]

    # convert V to a vector
    V = np.reshape(V, (n**2,))

    # compute the ssa and grad
    ss_absc, ss_absc_grad = ssa(A, eps)
    ss_absc_grad_flat = np.reshape(ss_absc_grad, (n**2,)) # one column vector

    # compute norms and dot products
    grad_norm = np.sqrt(np.dot(ss_absc_grad_flat, ss_absc_grad_flat))
    V_norm = np.sqrt(np.dot(V, V))
    V_dot_grad = np.dot(V, ss_absc_grad_flat)

    # find the amount which V is in the direction of the gradient
    grad_comp = V_dot_grad / np.power(grad_norm, 2)

    # find angle to the original from the tangent plane
    tangent_theta = np.arcsin(V_dot_grad / (grad_norm * V_norm))

    # remove the component in the direction of the gradient    
    V = V - (grad_comp * ss_absc_grad_flat)

    # renormalise and scale to step size multiplied by the square root of the dimension of the manifold
    new_V_norm = np.sqrt(np.dot(V, V))
    V = (step_size / new_V_norm) * np.sqrt(n**2 - 1) * V

    # reconvert to matrix shape:
    V = np.reshape(V, (n, n))

    # newton retraction:
    A = ssa_newton_retraction(A, V, eps, target)

    return A, tangent_theta

def mutate(A: np.array, eps: float, target: float, step_size: float=0.01):

    n = A.shape[0]    
    
    # generate a random vector
    V = np.random.random(size=(n, n))

    # tangent retract:
    A = tangent_retract(A, V, eps, target, step_size)[0]

    return A

def crossover(A: np.array, B: np.array, eps: float, target: float, iters: int=50):

    # compute the average of A and B
    C = (A + B) / 2.0

    # project this aveage to the manifold via repeated newton iterations
    C = optimize_ssa(C, eps, target, iters)

    return C
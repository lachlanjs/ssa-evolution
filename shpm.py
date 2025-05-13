""" 
Hurwitz-Schur Product Manifold
"""

import numpy as np
import pymanopt
from pymanopt.manifolds import Product, Stiefel, Grassmann, Euclidean
from pymanopt.manifolds.manifold import Manifold

class SHHessComplex2x2(Manifold):
    """ 
    Matrices of the form:
        | alpha, beta  |
        | gamma, alpha |
    where alpha < 0 and beta * gamma < 0    
    """

    def __init__(self):

        return
    

class SHHess(Manifold):

    def __init__(self):

        self.obo_idxs = []
        self.tbt_idxs = []

        return
    
    
    

class SHPM(Manifold):

    """ Schur-Hurwitz Product Manifold        
    """    

    def __init__(self, n: int):
        
        if n % 2 != 0:
            raise ValueError("need n to be divisible by 2")
        
        self.n = n

        name = f"Hurwitz-Schur Product Manifold of dimension {n}"
        dimension = n**2 + 1

        self.Q_stief_m = Stiefel(n, n)        
        self.T_D_euc_pm = Euclidean(n//2,2) # these guys need the exp(-x) treatment
        self.T_D_euc_pm
        self.T_U_euc_pm = Product([
            Euclidean(n-1-i,)
            for i in range(n-1)
        ])

        super().__init__(name, dimension)    
    
    def random_point(self):
        return (
            self.Q_stief_m.random_point(),
            self.T_D_euc_pm.random_point(),
            self.T_U_euc_pm.random_point()
        )
    
    def random_tangent_vector(self, P):
        return (
            self.Q_stief_m.random_tangent_vector(P[0]),            
            self.T_D_euc_pm.random_tangent_vector(P[1]),
            self.T_U_euc_pm.random_tangent_vector(P[2])
        )

    def exp(self, P: tuple[Manifold], T: tuple):
        return (
            self.Q_stief_m.exp(P[0], T[0]),            
            self.T_D_euc_pm.exp(P[1], T[1]),
            self.T_U_euc_pm.exp(P[2], T[2])
        )
    
    def mutate(self, P: tuple[Manifold], mag: float):
        """ 

        Args:
            P (tuple[Manifold]): _description_
            mag (float): _description_

        Returns:
            _type_: _description_
        """

        # NOTE: DON"T FORGET POSSIBLE PERMUTATION OF Q
        return self.exp(P, (
            self.Q_stief_m.random_tangent_vector(P[0]) * mag,            
            self.T_D_euc_pm.random_tangent_vector(P[1]) * mag,
            self.T_U_euc_pm.random_tangent_vector(P[2]) * mag            
        ))
    
    def crossover(self, P_1: tuple[Manifold], P_2: tuple[Manifold]):     
        """
        We just break all the rules here, because we're rebels
        - Q's are matrix multiplied
        - Grasmmanians are averaged on the unit sphere
            - Just take average and normalise with the vector representations that are closer together
        - Euclidean points are averaged
        """
        
        return (
            P_1[0] @ P_2[0],            
            self.T_D_euc.pair_mean(P_1[1], P_2[1]),
            self.T_U_euc.pair_mean(P_1[2], P_2[2]),            
        )
    
    def assemble(self, P):
        """ assemble a point on the manifold into a matrix

        Args:
            p (_type_): point on the HPM manifold
        """

        n = self.n
        T = np.zeros((n, n))

        # place the diagonal components
        for d_idx in range(0, n, 2):
            T[d_idx, d_idx + 1] = -np.exp(P[1][d_idx//2, 0])
            T[d_idx + 1, d_idx + 1] = -np.exp(P[1][(d_idx//2), 1])
            
        # place the strict upper triangular part
        for start_idx in range(1, n):
            for d_idx in range(n - start_idx):
                T[start_idx + d_idx][d_idx] = P[2][start_idx - 1][d_idx]            
        
        return P[0] @ T @ P[0].T
    
    # abstract method fun...
    def inner_product(self, P, T_A, T_B):

        # sum of the inner products in the product manifold:
        g_0 = self.Q_stief_m.inner_product(P[0], T_A[0], T_B[0])
        g_1 = self.T_D_euc_pm.inner_product(P[1], T_A[1], T_B[1])
        g_2 = self.T_U_euc_pm.inner_product(P[2], T_A[2], T_B[2])

        return g_0 + g_1 + g_2
    
    def norm(self, P, T):
        return self.inner_product(P, T, T)
    
    def projection(self, P, V):
        return (
            self.Q_stief_m.projection(P[0], V[0]),
            self.T_D_euc_pm.projection(P[1], V[1]),
            self.T_U_euc_pm.projection(P[2], V[2])
        )
    
    def zero_vector(self, P):
        return (
            self.Q_stief_m.projection(P[0]),
            self.T_D_euc_pm.projection(P[1]),
            self.T_U_euc_pm.projection(P[2])
        )
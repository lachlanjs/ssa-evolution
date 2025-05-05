""" 
Hurwitz-Schur Product Manifold
"""

import numpy as np
import pymanopt
from pymanopt.manifolds import Product, Stiefel, Grassmann, Euclidean
from pymanopt.manifolds.manifold import Manifold

class SHPM(Manifold):

    """ Schur-Hurwitz Product Manifold

        Points are represented as tuples:
            (Q, T_D_gra_ad, T_D_euc_ad, T_D_euc_bc, T_U_euc)
        Tangent Vectors are represented analoously

        The Schur-Hurwitz Product Manifold is defined as a product
        of Stiefel, Grassmann and Euclidean manifolds, designed to 
        (over)-parameterise the space of matrices with eigenvalues
        whose real part is < 0.0
    """    

    def __init__(self, n: int):
        
        if n % 2 != 0:
            raise ValueError("need n to be divisible by 2")

        name = f"Hurwitz-Schur Product Manifold of dimension {n}"
        dimension = n**2 + 1

        self.stief_m = Stiefel(n, n)
        self.T_D_gra_ad_pm = Grassmann(2, 1, n//2)
        self.T_D_euc_ad_pm = Product([
            Euclidean((1,)) for i in range(n//2)
        ])
        self.T_D_euc_bc_pm = Product([
            Euclidean((2,)) for i in range(n//2)
        ])
        self.T_U_euc_pm = Product([
            Euclidean((n-2-i,))
            for i in range(n-2)
        ])  

        super().__init__(name, dimension)    
    
    def random_point(self):
        return (
            self.stief.random_point(),
            self.T_D_gra_ad_pm.random_point(),
            self.T_D_euc_ad_pm.random_point(),
            self.T_D_euc_bc_pm.random_point(),
            self.T_U_euc_pm.random_point()
        )
    
    def random_tangent_vector(self, P):
        return (
            self.stief.random_tangent_vector(P[0]),
            self.T_D_gra_ad_pm.random_tangent_vector(P[1]),
            self.T_D_euc_ad_pm.random_tangent_vector(P[2]),
            self.T_D_euc_bc_pm.random_tangent_vector(P[3]),
            self.T_U_euc_pm.random_tangent_vector(P[4])
        )

    def exp(self, P: tuple[Manifold], T: tuple):
        return (
            self.stief_m.exp(P[0], T[0]),
            self.T_D_gra_ad_pm.exp(P[1], T[1]),
            self.T_D_euc_ad_pm.exp(P[2], T[2]),
            self.T_D_euc_bc_pm.exp(P[3], T[3]),
            self.T_U_euc_pm.exp(P[4], T[4])
        )
    
    def mutate(self, P: tuple[Manifold], mag: float):
        return self.exp(P, (
            self.stief_m.random_tangent_vector(P[0]) * mag,
            self.T_D_gra_ad_pm.random_tangent_vector(P[1]) * mag,
            self.T_D_euc_ad_pm.random_tangent_vector(P[2]) * mag,
            self.T_D_euc_bc_pm.random_tangent_vector(P[3]) * mag,
            self.T_U_euc_pm.random_tangent_vector(P[4] * mag)
        ))
    
    def crossover(self, P_1: tuple[Manifold], P_2: tuple[Manifold]):     
        """
        We just break all the rules here, because we're rebels
        - Q's are matrix multiplied
        - Grasmmanians are averaged on the unit sphere
            - Just take average and normalise
        - Euclidean points are averaged
        """
        
        return (
            P_1[0] @ P_2[0],

        )
    
    def assemble(self, P):
        """ assemble a point on the manifold into a matrix

        Args:
            p (_type_): point on the HPM manifold
        """

        return
    
    
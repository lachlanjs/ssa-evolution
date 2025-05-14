import numpy as np 
import pymanopt
from pymanopt.manifolds import Product, Stiefel, Euclidean
from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance

from random import random, sample

def rand_norm(n: int):
    v = np.random.normal(0.0, 1.0, size=(n,))    
    return v / np.linalg.norm(v)

# CONSTANTS
B_MERGE_MAG = 1.0

class HbergTangentVector(ndarraySequenceMixin):

    def __init__(self, n: int, t_UT, t_SD, t_HD):
        self.n = n
        self.t_UT = t_UT
        self.t_SD = t_SD
        self.t_HD = t_HD # list of np arrays

    def check_compatibility(self, other):
        # check conference of HD shapes and such
        if self.n != other.n:
            raise ValueError("The dimensionality n of the two tangent vectors must match")
        if len(self.t_HD) != len(other.t_HD):
            raise ValueError("The number of elements in the HD component of the tangent vectors is not equal")
        for hd_self, hd_other in zip(self.t_HD, other.t_HD):
            if hd_self.shape != hd_other.shape:
                raise ValueError("The shape of elements in the HD component of the tangent vectors must be the same")                
        return True

    # @return_as_class_instance(unpack=False)
    def __add__(self, other):                
        self.check_compatibility(other) # compability check

        return HbergTangentVector(
            self.n,
            self.t_UT + other.t_UT, 
            self.t_SD + other.t_SD, 
            [hd_self + hd_other for hd_self, hd_other in zip(self.t_HD, self.t_HD)]
        )

    # @return_as_class_instance(unpack=False)
    def __sub__(self, other):
        self.check_compatibility(other) # compability check

        return HbergTangentVector(
            self.n,
            self.t_Ut - other.t_UT,
            self.t_SD - other.t_SD, 
            [hd_self - hd_other for hd_self, hd_other in zip(self.t_HD, self.t_HD)]
        )

    # @return_as_class_instance(unpack=False)
    def __mul__(self, scalar):
        return HbergTangentVector(
            self.n,
            scalar * self.t_UT,
            scalar * self.t_SD,
            [scalar * hd for hd in self.t_HD]
        )

    __rmul__ = __mul__ # dunno what's going on here tbh

    # @return_as_class_instance(unpack=False)
    def __truediv__(self, scalar):
        return HbergTangentVector(
            self.n,
            self.t_UT / scalar,
            self.t_SD / scalar,
            [hd / scalar for hd in self.t_HD]
        )        

    # @return_as_class_instance(unpack=False)
    def __neg__(self):
        return HbergTangentVector(
            self.n,
            -self.t_UT,
            -self.t_SD,
            [-hd for hd in self.t_HD]
        )

class Hberg(Manifold):

    def __init__(self, 
        n: int, 
        HD_1x1_merge_chance: float = 0.1, 
        HD_2x2_split_chance: float = 0.1
    ):

        assert n > 0, "the dimensionality n should be positive"

        self.n = n

        self.HD_1x1_merge_chance = HD_1x1_merge_chance
        self.HD_2x2_split_chance = HD_2x2_split_chance

        self.UT = Product([Euclidean(n-i) for i in range(2, n)])
        self.SD = Euclidean(n-1)       

    def tangent_compatibility(self, p, t):

        if self.n != t.n:
            raise ValueError("Original dimensionality of Hberg manifold and tangent vector must match")
        
        # check hd:
        if len(p[2]) != len(t.t_HD):
            raise ValueError("The number of elements in the HD component of the tangent vector must match the point")
        
        for hd_self, hd_other in zip(p[2], t.t_HD):
            if hd_self[1].shape != hd_other.shape:
                raise ValueError("The shape of elements in the HD component of the tangent vectors must be the same")                

        return True
    
    def random_point(self):
        p_UT = self.UT.random_point()
        p_SD = self.SD.random_point()

        # construct a random set of eigenvalues for the diagonal
        # (type, values)
        p_HD = []
        d_idx = 0
        while d_idx < self.n:
            e_type = 1 if random() < 0.67 else 2 if d_idx < self.n - 1 else 1
            p_HD.append([e_type, np.random.normal(0.0, 1.0, size=(e_type,))])
            d_idx += e_type

        return (p_UT, p_SD, p_HD)    
    
    def assemble(self, p):
        n = self.n

        T = np.zeros(shape=(self.n, self.n))

        for d_idx in range(0, n-1):
            T[d_idx][1 + d_idx] = p[1][d_idx]

        for d_start_idx in range(2, n):
            for d_idx in range(0, n-d_start_idx):
                T[d_idx][d_start_idx + d_idx] = p[0][d_start_idx - 2][d_idx]
        
        d_idx = 0
        for hd in p[2]:
            
            if hd[0] == 1:  # real
                # add eigenvalue to the diagonal
                T[d_idx, d_idx] = -np.exp(hd[1])
            else:           # complex
                # ensure that the value opposite the superdiagonal is of opposite sign
                T[d_idx, d_idx] = -np.exp(hd[1][0])
                T[d_idx + 1, d_idx + 1] = -np.exp(hd[1][0])
                T[d_idx + 1, d_idx] = -1 * np.sign(p[1][d_idx]) * np.exp(hd[1][1])                

            d_idx += hd[0]

        return T
    
    def random_tangent_vector(self, p):
        return HbergTangentVector(
            self.n,
            self.UT.random_tangent_vector(p[0]),
            self.SD.random_tangent_vector(p[1]),
            [rand_norm(hd[1].size) for hd in p[2]]
        )
    
    def exp(self, p: tuple[Manifold], t: HbergTangentVector):  
        # check that shapes match:      
        self.tangent_compatibility(p, t)

        return (
            self.UT.exp(p[0], t.t_UT),
            self.SD.exp(p[1], t.t_SD),
            [[p_hd[0], p_hd[1] + t_hd] for p_hd, t_hd in zip(p[2], t.t_HD)]
        )
    
    def mutate(self, p, mag: float, verbose: bool = False):

        # move slightly in the direction of a random tangent vector
        ## get a random tangent vector
        ## rescale for fairness
        t = self.random_tangent_vector(p) * np.random.random() * mag        
        p = self.exp(p, t)

        # do stuff to the Hberg diagonal
        ## identify real eigenvalues which are next to eachother
        r_adj_idxs = []
        for hd_idx, hd in enumerate(p[2][:-1]):
            if hd[0] == 1 and p[2][hd_idx + 1][0] == 1:
                # pair
                r_adj_idxs.append(hd_idx)

        idx_idx = 0
        del_idxs = []
        while idx_idx < len(r_adj_idxs):
            r_adj_idx = r_adj_idxs[idx_idx]
            if random() < self.HD_1x1_merge_chance:
                # do the merge
                if verbose: print("merged")
                ## get the new values
                d = np.abs(p[2][r_adj_idx][1][0] - p[2][r_adj_idx + 1][1][0])

                a = 0.5 * (p[2][r_adj_idx][1][0] + p[2][r_adj_idx + 1][1][0])   # the real component
                b = 2.0 * (np.random.random() - 0.5) * d                        # the other value

                ## change the information in the first one
                p[2][r_adj_idx][0] = 2
                p[2][r_adj_idx][1] = np.array([a, b])

                ## remeber the idx of the second one to delete later
                del_idxs.append(r_adj_idx + 1)
                
                # ---
                # check if this affects the next pair of real eigenvalues that were identified
                if idx_idx != len(r_adj_idxs) - 1:
                    if r_adj_idx == r_adj_idxs[idx_idx + 1] - 1:
                        idx_idx += 1

            idx_idx += 1         

        # perform necessary deletions
        ## NOTE: the following works because the list is in increasing order
        del_offset = 0
        for del_idx in del_idxs:
            del p[2][del_idx - del_offset]
            del_offset += 1

        ## split apart a complex eigenvalue pair
        hd_idx = 0
        while hd_idx < len(p[2]):
            hd = p[2][hd_idx]
            if hd[0] == 1:
                hd_idx += 1
                continue
            
            if random() < self.HD_2x2_split_chance:
                if verbose: print("split")
                # splitty time
                real_1 = hd[1][0] + 2.0 * (np.random.random() - 0.5) * mag
                real_2 = hd[1][0] + 2.0 * (np.random.random() - 0.5) * mag
                
                # change the information of the first
                p[2][hd_idx] = [1, np.array([real_1])]
    
                # insert the second
                p[2].insert(hd_idx + 1, [1, np.array([real_2])])
    
                hd_idx += 1 # NOTE: I think                

            hd_idx += 1
                    
        return p

    def crossover(self, p_a: tuple[Manifold], p_b: tuple[Manifold], verbose: bool=False):

        HD_c = []
        d_idx = 0        

        def create_HD_str(p_hd):
            HD_str = []
            HD_idxs = []
            for hd_idx, hd in enumerate(p_hd):
                if hd[0] == 1:
                    HD_str.append(1)
                    HD_idxs.append(hd_idx)
                else:
                    HD_str.append(2)
                    HD_str.append(0)
                    HD_idxs.append(hd_idx)
                    HD_idxs.append(hd_idx)

            return HD_str, HD_idxs

        HD_a_str, HD_a_idxs = create_HD_str(p_a[2])
        HD_b_str, HD_b_idxs = create_HD_str(p_b[2])

        while d_idx < self.n:    
            
            hd_a_idx = HD_a_idxs[d_idx]
            hd_b_idx = HD_b_idxs[d_idx]
            
            if HD_a_str[d_idx] == 0:
                # have to pick b
                HD_c.append(p_b[2][hd_b_idx])
                d_idx += p_b[2][hd_b_idx][0]
            elif HD_b_str[d_idx] == 0:
                # have to pick a
                HD_c.append(p_a[2][hd_a_idx])
                d_idx += p_a[2][hd_a_idx][0]                 
            elif HD_a_str[d_idx] == 1 and HD_b_str[d_idx] == 1:
                # take the mean
                HD_c.append((1, 0.5 * (p_a[2][hd_a_idx][1] + p_b[2][hd_b_idx][1])))
                d_idx += 1                
            elif HD_a_str[d_idx] == 2 and HD_b_str[d_idx] == 2:
                # take the mean
                HD_c.append((2, 0.5 * (p_a[2][hd_a_idx][1] + p_b[2][hd_b_idx][1])))
                d_idx += 2                
            else:                    
                # pick one at random
                if random() < 0.5:
                    # pick a
                    HD_c.append(p_a[2][hd_a_idx])
                    d_idx += p_a[2][hd_a_idx][0]                    
                else:
                    # pick b
                    HD_c.append(p_b[2][hd_b_idx])
                    d_idx += p_b[2][hd_b_idx][0]                    

        return (
            self.UT.pair_mean(p_a[0], p_b[0]),
            self.SD.pair_mean(p_a[1], p_b[1]),
            HD_c
        )
    
    # abstract method joy...    
    def inner_product(self, p, t_a: HbergTangentVector, t_b: HbergTangentVector):

        return
    
    def norm(self, p, t: HbergTangentVector):

        return
    
    def projection(self, p, v):

        return
    
    def zero_vector(self, p):

        return


# NOTE do we need a tangent for this thing too?

# constants
CRAZY_CROSSOVER_CHANCE = 0.1

class SHPM(Manifold):

    def __init__(self, n: int, Q_swap_chance: float=0.1, crazy_Q_xv_chance: float = 0.05):

        assert n > 0, "the dimensionality n should be positive"
        self.n = n
        self.Q_swap_chance = Q_swap_chance
        self.crazy_Q_xv_chance = crazy_Q_xv_chance

        self.stiefel = Stiefel(n, n)
        self.hberg = Hberg(n)        

        return
    
    def random_point(self):        
        return (
            self.stiefel.random_point(),
            self.hberg.random_point()
        )
    
    def random_tangent_vector(self, p):
        return (
            self.stiefel.random_tangent_vector(p[0]),
            self.hberg.random_tangent_vector(p[1])
        )
    
    def assemble(self, p):        
        return p[0] @ self.hberg.assemble(p[1]) @ p[0].T
    
    def mutate(self, p, mag: float, verbose: bool = False):

        p_Q = self.stiefel.exp(p[0], self.stiefel.random_tangent_vector(p[0]) * random() * mag)
        # possible random permutation
        if random() < self.Q_swap_chance:

            # pick random indices without replacement:
            idx_1, idx_2 = sample(range(self.n), 2)
            if verbose: print(f"swapped: {idx_1} <-> {idx_2}")

            # switch em
            q1 = np.copy(p[0][:, idx_1])
            p[0][:, idx_1] = np.copy(p[0][:, idx_2])
            p[0][:, idx_2] = q1

        p_H = self.hberg.mutate(p[1], mag, verbose)

        return (p_Q, p_H)
    
    def crossover(self, p_a, p_b):

        p_Q = p_a[0] @ p_b[0] if random() < self.crazy_Q_xv_chance else p_a[0] if random() < 0.5 else p_b[0]                     
        p_H = self.hberg.crossover(p_a[1], p_b[1])

        return (p_Q, p_H)
    
    # abstract method joy...    
    def inner_product(self, p, t_a, t_b):

        return
    
    def norm(self, p, t):

        return
    
    def projection(self, p, v):

        return
    
    def zero_vector(self, p):

        return


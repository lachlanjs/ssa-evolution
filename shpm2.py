import numpy as np 
import pymanopt
from pymanopt.manifolds import Product, Stiefel, Euclidean
from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools import ndarraySequenceMixin, return_as_class_instance

from scipy.optimize import linear_sum_assignment

from random import random, sample

def rand_norm(n: int):
    v = np.random.normal(0.0, 1.0, size=(n,))    
    return v / np.linalg.norm(v)

def posify(x):    
    return np.where(x < 0, np.exp(x), x + 1)

def squish_distance(x):
    return np.where(x>0, np.power(x, 2.0) / (1 + np.power(x, 2.0)), 0.0)

# CONSTANTS
B_MERGE_MAG = 1.0
MAX_EIG_SWAPS = 5
MAX_Q_SWAPS = 5

class HbergTangentVector(ndarraySequenceMixin):

    def __init__(self, n: int, t_UT, t_HD):
        self.n = n
        self.t_UT = t_UT        
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
            [hd_self + hd_other for hd_self, hd_other in zip(self.t_HD, self.t_HD)]
        )

    # @return_as_class_instance(unpack=False)
    def __sub__(self, other):
        self.check_compatibility(other) # compability check

        return HbergTangentVector(
            self.n,
            self.t_UT - other.t_UT,            
            [hd_self - hd_other for hd_self, hd_other in zip(self.t_HD, self.t_HD)]
        )

    # @return_as_class_instance(unpack=False)
    def __mul__(self, scalar):
        return HbergTangentVector(
            self.n,
            scalar * self.t_UT,            
            [scalar * hd for hd in self.t_HD]
        )

    __rmul__ = __mul__ # dunno what's going on here tbh

    # @return_as_class_instance(unpack=False)
    def __truediv__(self, scalar):
        return HbergTangentVector(
            self.n,
            self.t_UT / scalar,            
            [hd / scalar for hd in self.t_HD]
        )        

    # @return_as_class_instance(unpack=False)
    def __neg__(self):
        return HbergTangentVector(
            self.n,
            -self.t_UT,            
            [-hd for hd in self.t_HD]
        )

class Hberg(Manifold):

    def __init__(self, 
        n: int,
        UT_mut_mag: float = 1.0,        
        HD_1x1_merge_chance: float = 0.15, 
        HD_2x2_split_chance: float = 0.15,
        HD_eig_swap_chance: float = 0.33,
        HD_mut_mag: float = 1.0
    ):

        assert n > 0, "the dimensionality n should be positive"

        self.n = n
        self.UT_mut_mag = UT_mut_mag        
        self.HD_1x1_merge_chance = HD_1x1_merge_chance
        self.HD_2x2_split_chance = HD_2x2_split_chance
        self.HD_eig_swap_chance = HD_eig_swap_chance
        self.HD_mut_mag = HD_mut_mag

        # self.UT = Product([Euclidean(n-i) for i in range(2, n)])        
        self.UT = Euclidean(n, n)

    def tangent_compatibility(self, p, t):

        if self.n != t.n:
            raise ValueError("Original dimensionality of Hberg manifold and tangent vector must match")
        
        # check hd:
        if len(p[1]) != len(t.t_HD):
            raise ValueError("The number of elements in the HD component of the tangent vector must match the point")
        
        for hd_self, hd_other in zip(p[1], t.t_HD):
            if hd_self[1].shape != hd_other.shape:
                raise ValueError("The shape of elements in the HD component of the tangent vectors must be the same")                

        return True
    
    def random_point(self):
        p_UT = self.UT.random_point()        

        # construct a random set of eigenvalues for the diagonal
        # (type, values)
        p_HD = []
        d_idx = 0
        while d_idx < self.n:
            e_type = 1 if random() < 0.67 else 2 if d_idx < self.n - 1 else 1
            p_HD.append([e_type, np.random.normal(0.0, 1.0, size=(e_type,))])
            d_idx += e_type

        return (p_UT, p_HD)    
    
    def assemble(self, p):
        n = self.n

        # T = np.zeros(shape=(self.n, self.n))
        T = np.triu(p[0], k=1)                
        
        d_idx = 0
        for hd in p[1]:
            
            if hd[0] == 1:  # real
                # add eigenvalue to the diagonal
                T[d_idx, d_idx] = -posify(hd[1])
            else:           # complex
                # ensure that the value opposite the superdiagonal is of opposite sign
                T[d_idx, d_idx] = -posify(hd[1][0])
                T[d_idx + 1, d_idx + 1] = -posify(hd[1][0])
                T[d_idx + 1, d_idx] = -1 * np.sign(T[d_idx][d_idx + 1]) * posify(hd[1][1])                                

            d_idx += hd[0]

        return T
    
    def random_tangent_vector(self, p):
        return HbergTangentVector(
            self.n,
            self.UT.random_tangent_vector(p[0]),            
            [rand_norm(hd[1].size) for hd in p[1]]
        )
    
    def exp(self, p: tuple[Manifold], t: HbergTangentVector):  
        # check that shapes match:      
        self.tangent_compatibility(p, t)

        return (
            self.UT.exp(p[0], t.t_UT),            
            [[p_hd[0], p_hd[1] + t_hd] for p_hd, t_hd in zip(p[1], t.t_HD)]
        )
    
    def mutate(self, p, mag: float, verbose: bool = False):

        # move slightly in the direction of a random tangent vector
        ## get a random tangent vector
        ## rescale for fairness
        t = self.random_tangent_vector(p) * mag

        # the following is the same as exp but it rescales stuff 
        ## p[0] = self.UT.exp(p[0], t.t_UT * self.UT_mut_mag)
        ## p[1] = [[p_hd[0], p_hd[1] + t_hd * self.HD_mut_mag] for p_hd, t_hd in zip(p[1], t.t_HD)]

        p = (
            self.UT.exp(p[0], t.t_UT * np.random.normal() * self.UT_mut_mag * self.n),
            [[p_hd[0], p_hd[1] + t_hd * np.random.normal() * self.HD_mut_mag] for p_hd, t_hd in zip(p[1], t.t_HD)]
        )

        # p = self.exp(p, t)

        # do stuff to the Hberg diagonal
        ## identify real eigenvalues which are next to eachother
        r_adj_idxs = []
        for hd_idx, hd in enumerate(p[1][:-1]):
            if hd[0] == 1 and p[1][hd_idx + 1][0] == 1:
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
                d = np.abs(p[1][r_adj_idx][1][0] - p[1][r_adj_idx + 1][1][0])

                a = 0.5 * (p[1][r_adj_idx][1][0] + p[1][r_adj_idx + 1][1][0])   # the real component
                b = 2.0 * np.random.normal() * mag                              # the other value

                ## change the information in the first one
                p[1][r_adj_idx][0] = 2
                p[1][r_adj_idx][1] = np.array([a, b])

                ## remember the idx of the second one to delete later
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
            del p[1][del_idx - del_offset]
            del_offset += 1

        ## split apart a complex eigenvalue pair
        hd_idx = 0
        while hd_idx < len(p[1]):
            hd = p[1][hd_idx]
            if hd[0] == 1:
                hd_idx += 1
                continue
            
            if random() < self.HD_2x2_split_chance:
                if verbose: print("split")
                # splitty time
                real_1 = hd[1][0] + np.random.normal() * mag
                real_2 = hd[1][0] + np.random.normal() * mag
                
                # change the information of the first
                p[1][hd_idx] = [1, np.array([real_1])]
    
                # insert the second
                p[1].insert(hd_idx + 1, [1, np.array([real_2])])
    
                hd_idx += 1 # NOTE: I think...              

            hd_idx += 1

        ## spectrum swapping
        for _ in range(MAX_EIG_SWAPS):
            if random() < self.HD_eig_swap_chance:            
                # pick idxs
                idx_1, idx_2 = sample(range(len(p[1])), 2)
                p[1][idx_1], p[1][idx_2] = p[1][idx_2], p[1][idx_1]
            else: break
                    
        return p
    
    def get_eigenspectrum(self, p):

        """ extract eigenvectors from the hessenberg matrix diagonal
        """

        eigs_real = []
        eigs_comp = []

        d_idx = 0
        for hd_idx, hd in enumerate(p[1]):

            if p[1][hd_idx][0] == 1: # real
                eigs_real.append(hd[1][0])
                d_idx += 1
            else:                
                im = np.sqrt(np.abs(hd[1][1] * p[0][d_idx, d_idx + 1]))
                # print(f"hd: {hd}, im: {im}")
                eigs_comp.append(hd[1][0] + im * 1j)
                eigs_comp.append(hd[1][0] - im * 1j)
                d_idx += 2

        # print(f"eigs_real: {eigs_real}")
        # print(f"eigs_comp: {eigs_comp}") 

        return np.array(eigs_real + eigs_comp)
    
    # helper
    def create_HD_str(self, p_hd):
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

    def crossover(self, p_a: tuple[Manifold], p_b: tuple[Manifold], verbose: bool=False):

        HD_c = []
        d_idx = 0                

        HD_a_str, HD_a_idxs = self.create_HD_str(p_a[1])
        HD_b_str, HD_b_idxs = self.create_HD_str(p_b[1])

        while d_idx < self.n:    
            
            hd_a_idx = HD_a_idxs[d_idx]
            hd_b_idx = HD_b_idxs[d_idx]
            
            if HD_a_str[d_idx] == 0:
                # have to pick b
                HD_c.append(p_b[1][hd_b_idx])
                d_idx += p_b[1][hd_b_idx][0]
            elif HD_b_str[d_idx] == 0:
                # have to pick a
                HD_c.append(p_a[1][hd_a_idx])
                d_idx += p_a[1][hd_a_idx][0]                 
            elif HD_a_str[d_idx] == 1 and HD_b_str[d_idx] == 1:
                # take the mean
                HD_c.append((1, 0.5 * (p_a[1][hd_a_idx][1] + p_b[1][hd_b_idx][1])))
                d_idx += 1                
            elif HD_a_str[d_idx] == 2 and HD_b_str[d_idx] == 2:
                # take the mean
                HD_c.append((2, 0.5 * (p_a[1][hd_a_idx][1] + p_b[1][hd_b_idx][1])))
                d_idx += 2                
            else:                    
                # pick one at random
                if random() < 0.5:
                    # pick a
                    HD_c.append(p_a[1][hd_a_idx])
                    d_idx += p_a[1][hd_a_idx][0]
                else:
                    # pick b
                    HD_c.append(p_b[1][hd_b_idx])
                    d_idx += p_b[1][hd_b_idx][0]                    

        return (
            self.UT.pair_mean(p_a[0], p_b[0]),            
            HD_c
        )
    
    def speciation_distance(self, p_a, p_b):

        # distance between upper triangular components
        ## NOTE: this thing appears to scale with n
        UT_dis = np.linalg.norm(np.triu(p_a[0], k=1) - np.triu(p_b[0], k=1)) / self.n

        # distance between diagonal components
        # if the diagonal component does not match, then add 1
        # if the diagonal component does match, then add a f
        HD_a_str, HD_a_idxs = self.create_HD_str(p_a[1])
        HD_b_str, HD_b_idxs = self.create_HD_str(p_b[1])

        d_idx = 0
        HD_dis = 0.0
        while d_idx < self.n:
            if HD_a_str[d_idx] != HD_b_str[d_idx]:
                HD_dis += 1.0
            else:
                HD_a_idx = HD_a_idxs[d_idx]
                HD_b_idx = HD_b_idxs[d_idx]

                if HD_a_str[d_idx] == 0:
                    HD_dis += squish_distance(np.abs(p_a[1][HD_a_idx][1][1] - p_b[1][HD_b_idx][1][1]))
                else:
                    HD_dis += squish_distance(np.abs(p_a[1][HD_a_idx][1][0] - p_b[1][HD_b_idx][1][0]))
            
            d_idx += 1

        HD_dis /= self.n

        return UT_dis, HD_dis

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

    def __init__(self, 
        n: int, 
        Q_mut_mag: float=1.0,
        Q_swap_chance: float=0.1, 
        crazy_Q_xv_chance: float = 0.05,
        Q_sep: float = 1.0,
        UT_sep: float = 1.0,
        HD_sep: float = 1.0,
        **hberg_params
    ):

        assert n > 0, "the dimensionality n should be positive"
        self.n = n
        self.Q_mut_mag = Q_mut_mag
        self.Q_swap_chance = Q_swap_chance
        self.crazy_Q_xv_chance = crazy_Q_xv_chance

        self.Q_sep = Q_sep
        self.UT_sep = UT_sep
        self.HD_sep = HD_sep

        self.stiefel = Stiefel(n, n)
        self.hberg = Hberg(n, **hberg_params)

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

        p_Q = self.stiefel.exp(p[0], 
            self.stiefel.random_tangent_vector(p[0]) * np.random.normal() * mag * self.Q_mut_mag
        )

        # possible random permutation
        for _ in range(MAX_Q_SWAPS):
            if random() < self.Q_swap_chance:

                # pick random indices without replacement:
                idx_1, idx_2 = sample(range(self.n), 2)
                if verbose: print(f"swapped: {idx_1} <-> {idx_2}")
                # switch em
                p[0][:, [idx_1, idx_2]] = p[0][:, [idx_2, idx_1]]
            else: break

        p_H = self.hberg.mutate(p[1], mag, verbose)

        return (p_Q, p_H)
    
    def crossover(self, p_a, p_b):

        p_Q = p_a[0] @ p_b[0] if random() < self.crazy_Q_xv_chance else p_a[0] if random() < 0.5 else p_b[0]                     
        p_H = self.hberg.crossover(p_a[1], p_b[1])

        return (p_Q, p_H)
    
    def Q_distance(self, p_a, p_b):

        eigs = np.linalg.eigvals(p_a[0].T @ p_b[0])

        Q_dis = np.linalg.norm(np.angle(eigs))

        return Q_dis

    def same_species(self, p_a, p_b):

        eigs = np.linalg.eigvals(p_a[0].T @ p_b[0])

        Q_dis = np.linalg.norm(np.angle(eigs) / np.pi)

        UT_dis, HD_dis = self.hberg.speciation_distance(p_a[1], p_b[1])

        if Q_dis > self.Q_sep:
            return False
        if UT_dis > self.UT_sep:
            return False
        if HD_dis > self.HD_sep:
            return False
        
        return True
    
    def same_species_eigen(self, p_a, p_b):

        eigen_a = []
        eigen_b = []

        

        return
    
    # abstract method joy...    
    def inner_product(self, p, t_a, t_b):

        return
    
    def norm(self, p, t):

        return
    
    def projection(self, p, v):

        return
    
    def zero_vector(self, p):

        return


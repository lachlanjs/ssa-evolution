import numpy as np


from random import random, sample
from numpy.random import choice

from copy import deepcopy

from tqdm import tqdm

def evo_alg(
    p: list,
    m: int, l: int, 
    # NOTE: the objective may have to include the code for assembling the phenotype from the genotype
    objective: callable, 
    mutate: callable, crossover: callable, same_species: callable,
    s: float = 0.1, c: float = 0.1,
    maxiter: int=10, eps: float=1e-3,
    progress_bar: bool = False
):
    """ runs a (mu, lambda) evolutionary algorithm (elites are not protected)

    Args:
        p (list): the initial population of candidates (should be mu, lambda but this will inevitably be corrected)
        m (int): the number of parents (those selected for the next generation)
        l (int): the number of offspring (added to the pool in the next generation so that there are always m+l in total)
        objective (callable): the objective function to be minimized, it is assumed that 0.0 is the minimum
        mutate (callable): the mutation operator. takes one candidate and a magnitude and produces a new one which is slightly mutated
        crossover (callable): the crossover operator. takes two candidates and produces a new one using their information
        same_species (callable): returns true if the two input candidates are of the same species
        s (float): the mutation step size
        c (float): the crossover rate (between 0 and 1) (represents the probability at which an offspring will be made using crossover)        
        maxiter (int): the maximum number of iterations to compute
        eps (float): the threshold distance to the target value of 0.0 within which to terminate

    Returns:
        _type_: _description_
    """

    population_objectives = []

    p = [[candidate, 0.0] for candidate in p]
    n_sames = []
    
    for iter_idx in range(maxiter):

        # evaluate candidates, create tuples where the second element is the objective function value
        for candidate_idx, (candidate, _) in enumerate(p): # NOTE: we may assume that the objective function is stochastic and thus might reorder the elites...
            p[candidate_idx][1] = objective(candidate)

        # sort candidates based on the objective function value in ascending order (best (lowest objective value) first)        
        p = sorted(p, key=lambda v: v[1])

        # add evaluation values to a list for tracking
        population_objectives.append([v for _, v in p])

        # check if the target value has been reached, 
        if p[0][1] < eps:
            break

        # NOTE: could add the calculation of the preference weights
        ## species based fitness:
        n_same_matrix = np.zeros(shape=(len(p), len(p)))
        
        for candidate_idx, (candidate, _) in enumerate(p):
            # find how many in the same species:            

            for candidate_idx_2, (candidate_2, _) in enumerate(p[candidate_idx + 1:]):             
                if same_species(candidate, candidate_2):
                    n_same_matrix[candidate_idx, candidate_idx_2] += 1
        
        for candidate_idx, (candidate, raw_fitness) in enumerate(p):
            p[candidate_idx][1] = raw_fitness / (1 + np.sum(n_same_matrix[candidate_idx, :]) + np.sum(n_same_matrix[:, candidate_idx]))

        n_sames.append(np.sum(n_same_matrix))

        # using the top [:m], generate a new list of candidates
        new_ps = []
        for new_candidate_idx in range(l):
            new_candidate = None
            # NOTE: could add weighting based on fitnessx   
            if random() < c:
                # crossover two candidates to get the new one
                new_candidate = crossover(*[parent[0] for parent in sample(p[:m], 2)])
            else:
                # just mutate           
                parent = deepcopy(sample(p[:m], 1)[0][0])
                new_candidate = mutate(parent, mag=s)

            new_ps.append([new_candidate, 0.0])
        
        p=new_ps


    return p[0][0], p[0][1], iter_idx, population_objectives, n_sames # best solution, its function value, #iterations, objective values over time


def generate_candidates():

    return


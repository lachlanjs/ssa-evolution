import numpy as np


from random import random, sample, choice
# from numpy.random import choice

from copy import deepcopy

from tqdm import tqdm

def find_species(
    candidate, 
    reps: dict, 
    same_species: callable
):    

    for s_idx, rep in reps.items():
        if same_species(candidate, reps): return s_idx

    return -1 # new species required

def rank_mapping(population):

    """ this function creates a rank ordering of the candidates based on their raw fitness

    Returns:
        dict: a dict keyed by (species_idx, candidate_idx) which yields that members rank
    """

    # dict is keyed by species and then by index
    collapse = []
    for species_idx, species in population.items():
        collapse += [
            (species_idx, candidate_idx, candidate, fitness) 
             for candidate_idx, (candidate, fitness) 
             in enumerate(species)
        ]
        
    # create mapping 
    collapse.sort(key= lambda elt: elt[3])

    rank_mapping = dict(
        ((species_idx, candidate_idx), fitness_rank)
        for fitness_rank, (species_idx, candidate_idx, _, _) in enumerate(collapse)
    )    

    return rank_mapping

def evo_alg_speciated(
    seed, # initial candidate
    population_size: int,    
    objective: callable, 
    mutate: callable, crossover: callable, 
    same_species: callable,
    get_norm_mapping: callable = None,
    r: float = 0.3, s: float = 0.1, c: float = 0.3,
    maxiter: int=10, eps: float=1e-3,
    progress_bar: bool = False
):
    
    # NOTE: add species population sizes from each iteration
    
    # create initial population from the initial candidate
    population = {0: [[deepcopy(seed), 0.0]]}
    raw_fitnesses = []

    reps = {0: population[0][0]}

    next_species_idx = 1
    idx = 0
    while idx < population_size - 1:
        
        # select a random member:
        s_idx = choice(population.keys())
        parent = deepcopy(choice(population[s_idx]))

        offspring = mutate(parent, s)

        # find which species it belongs to:
        offspring_s_idx = find_species(offspring, reps, same_species)

        if offspring_s_idx == -1: # next species
            population[next_species_idx] = [[offspring, 0.0]]
            reps[next_species_idx] = offspring
            next_species_idx += 1
        else: # existing species
            population[offspring_s_idx].append([offspring, 0.0])

        idx += 1

    for iter_idx in range(maxiter):

        # evalaute fitness of population members
        # also adjust fitnesses based on the size of each species    
        raw_fitnesses_ = []
        for species_idx, species in population.items():
            species_size = len(species)
            for candidate_idx, (candidate, _) in enumerate(species):
                fitness = objective(candidate)
                species[candidate_idx][1] = fitness
                raw_fitnesses_.append(fitness)

        # save raw fitnesses 
        raw_fitnesses.append(raw_fitnesses)

        # check if any have reached the solution:
        # NOTE fill this in optionally
        
        # adjust fitnesses through optional normalisation
        if get_norm_mapping:
            norm_mapping = get_norm_mapping(population)

            population = {
                species_idx: [
                    [candidate, norm_mapping[(species_idx, candidate_idx)]]
                    for candidate_idx, (candidate, _) in species
                ]
                for species_idx, species in population.items()
            }
        
        # divide by the number of candidates in each species:
        population = {
            species_idx: [
                [candidate, fitness / len(species)]
                for candidate_idx, (candidate, fitness) in species
            ]
            for species_idx, species in population.items()
        }

        # calculate the mean adjusted fitness
        mean_adj_fitness = 0.0
        total = 0
        for species in population.values():
            for _, fitness in species:
                mean_adj_fitnes += fitness
                total += 1

        mean_adj_fitness /= total

        # calculate the new number of members of each species:
        species_N_offspring = {
            species_idx: sum([fitness for _, fitness in species]) / len(species)
            for species_idx, species in population
        }
        N_sum = sum(N_offspring.values())
        species_N_offspring = {
            species_idx: int(np.round((x * population_size) / N_sum))
            for species_idx, x in species_N_offspring.items()
        }
        
        

        # create the new population:
        ## only keep the top r quantile
        r_quantiles = {
            species_idx: np.quantile([fitness for _, fitness in species], 1.0-r)
            for species_idx, species in population
        }
        population = {
            species_idx: [
                elt for elt in species if elt[1] >= r_quantiles[species_idx]
                for species_idx, species in population
            ]
        }

        ## choose representatives for each species:
        reps = {
            species_idx: choice(species)[0]
            for species_idx, species in population.items()
        }
        new_population = {}
        for species_origin_idx, N_offspring in species_N_offspring.items():
            species_origin = population[species_origin_idx]            
            for offspring_idx in range(N_offspring):

                new_candidate = crossover(*[parent[0] for parent in sample(species_origin, k=2)])\
                    if random() < c else mutate(sample(species_origin, k=1)[0][0], s)
                
                # go through species in order to see which one the new candidate belongs to
                new_species_idx = find_species(new_candidate, reps, same_species)

                if new_species_idx == -1:
                    # create a new species
                    new_population[next_species_idx] = [(new_candidate, 0.0)]
                    next_species_idx += 1
                    continue

                if new_species_idx in new_population.keys():
                    new_population[new_species_idx].append([new_candidate, 0.0])
                else:
                    new_population[new_species_idx] = [(new_candidate, 0.0)]                
                

        # replace old population with new
        population = new_population
                
    return population, raw_fitnesses, 

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


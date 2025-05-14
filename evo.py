import numpy as np


def evo_alg(
    p: list,
    m: int, l: int, 
    objective: callable,
    mutation: callable, crossover: callable,
    s: float = 0.1, c: float = 0.1,
    maxiter: int=10, eps: float=1e-3
):
    """ runs a (mu + lambda) evolutionary algorithm (elites are not protected)

    Args:
        p (list): the initial population of candidates (should be m + lambda but this will inevitably be corrected)
        m (int): the number of parents (those selected for the next generation)
        l (int): the number of offspring (added to the pool in the next generation so that there are always m+l in total)
        objective (callable): the objective function to be minimized, it is assumed that 0.0 is the minimum
        mutation (callable): the mutation operator. takes one candidate and a magnitude and produces a new one which is slightly mutated
        crossover (callable): the crossover operator. takes two candidates and produces a new one using their information
        s (float): the mutation step size
        c (float): the crossover rate (between 0 and 1) (represents the probability at which an offspring will be made using crossover)        
        maxiter (int): the maximum number of iterations to compute
        eps (float): the threshold distance to the target value of 0.0 within which to terminate

    Returns:
        _type_: _description_
    """

    population_objectives = []

    p = [(candidate, 0.0) for candidate in p]
    
    for iter_idx in range(maxiter):

        # evaluate candidates, create tuples where the second element is the objective function value
        for candidate_idx, (candidate, _) in enumerate(p): # NOTE: we may assume that the objective function is stochastic and thus might reorder the elites...
            p[candidate_idx][1] = objective(candidate)

        # sort candidates based on the objective function value in ascending order (best (lowest objective value) first)
        p = sorted(p, key=lambda _, v: v)

        # add evaluation values to a list for tracking
        population_objectives.append([v for _, v in p])

        # check if the target value has been reached, 
        if p[0][1] < eps:
            break

        # using the top [:m], generate a new list of candidates
        # new_ps
        # for new_candidate_idx in range()

        # [:m].append(new_candidates)


        # repeat


    return p[0][0], p[0][1], iter_idx, population_objectives # best solution, its function value, #iterations, objective values over time


def generate_candidates():

    return


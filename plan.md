# .plan

- [x] Function to find preferred initial conditions for maximising the evoked energy
- [x] Write evolutionary algorithm
    - [x] Test with trying to directly match a target matrix
        It converges, but not exactly to the target, so there is some room here to keep testing. I need to compile all of the hyperparameters and then do a hyperparameter experiment for different values of $n$.
            - *Probably forget crossover?*
            - 1x1_merge_chance
            - 2x2_split_chance
            - Magnitude of mutation on UT
            - Magnitude of mutation on SD
            - Magnitude of mutation on HD
            - Magnitude of mutation on Q
            - *those four would be the most important*
            - Q_swap_chance
        - [ ] How many iterations compared to euclidean evolution?
        - [ ] Use this to determine suitable parameters for evolution on the manifolds
- [ ] Write objective function for the time until divergence from a target output for the preferred initial state
    - [ ] Run it on the manifold
    - [ ] Run it on the euclidean
- [ ] Create the ssa manifold formally

# take a step back

*why is this important?*

Stability Optimised Circuits are great and have interesting properties, as indicated by the Hennequin14 paper. They are achieved by optimising the ssa of a relatively random initial circuit. In terms of a goal, it is shown that they are capable of complex transient behaviour that can be mapped down to a variety of target outputs, including the snake and the butterly patterns.

Perhaps organisms start with an initially random circuit, stabilise it, and then learn readout weights.

What if the structure of the initial circuit was more sophisticated? What if biological evolution had optimised that initial circuit for subsequent learning in a more complex way? This would mean a more complicated objective function. Maybe a non-differentiable one. Let's say we have some complex non-differentiable function $f^*$ in mind, then we want to know what optimal solutions for this function look like, and we are not going to get to them comptuationally by using gradient based methods, because one is not available. Maybe we could pick a different objective function $g$, or guess at a variety of functions $g_1, ..., g_m$ such that optimising them could imply good performance on $f$, but the link is likely to be tenuous, difficult to explain, and we can never really know whether there are much better solutions to $f$. The smoothed spectral abscissa is an example of such a $g$.

An alternative to optimising surrogates like these speculated $g$ is to rethink the search space.  

Again, the point is not to find the biologically plausible optimisation scheme which got our real brains to work so well, it's to at least show what good solutions are when they exist, and we can cheat by reworking the search space, such as by using differential geometry/manifolds to parameterise a more useful / edifying set of matrices.

More interesting objective functions than the toy ones studied could include ability to stay stable under certain hebbian learning regimes, which is famously good at blowing things up to infinity - what if that is only true for certain circuits? That would be a really hard thing to find a differentiable objective for.


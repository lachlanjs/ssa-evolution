# .plan

- [ ] Function to find preferred initial conditions for maximising the evoked energy
- [ ] Write evolutionary algorithm
    - [ ] Test with trying to directly match a target matrix
        - [ ] How many iterations compared to euclidean evolution?
        - [ ] Use this to determine suitable parameters for evolution on the manifolds
- [ ] Write objective function for the time until divergence from a target output for the preferred initial state
    - [ ] Run it on the manifold
    - [ ] Run it on the euclidean
    - [ ] Get spectral abscissa back in there
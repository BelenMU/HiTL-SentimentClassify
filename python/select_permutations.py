import numpy as np
from itertools import permutations

def select_permutations(N_prev, top_permutation, max_permutations):
    """
    Generate a set of selected permutations based on the logic of the MATLAB function.
    This version ensures ordering matches MATLAB.
    """

    num_perms = np.math.factorial(N_prev + 1)
    
    if num_perms <= max_permutations:
        selected_perms = np.array(list(permutations(range(1, N_prev + 2))))
        return selected_perms  
    
    generated_permutations = np.vstack([top_permutation.copy() for _ in range(N_prev + 1)])

    for ii in range(N_prev):
        generated_permutations[ii + 1, ii] = top_permutation[ii + 1]
        generated_permutations[ii + 1, ii + 1] = top_permutation[ii]

    # Remove duplicate rows and sort
    permutations_set = np.unique(generated_permutations, axis=0)
    permutations_set = permutations_set[np.lexsort(np.rot90(permutations_set))]

    remaining_slots = max_permutations - len(permutations_set)
    if remaining_slots > 0:
        if num_perms <= 1e3:
            ranking_options = np.array(list(permutations(range(1, N_prev + 2))))
            current_set = {tuple(r) for r in permutations_set}
            remaining_perms = np.array([row for row in ranking_options if tuple(row) not in current_set])

            # Select the first `remaining_slots` permutations instead of random ones
            sampled_perms = remaining_perms[:min(remaining_slots, len(remaining_perms)), :]
            selected_perms = np.vstack([permutations_set, sampled_perms])
        else:
            while len(permutations_set) < max_permutations:
                sorted_perm = np.arange(1, N_prev + 2)
                if tuple(sorted_perm) not in {tuple(row) for row in permutations_set}:
                    permutations_set = np.vstack([permutations_set, sorted_perm])

            selected_perms = permutations_set
    else:
        selected_perms = permutations_set[:max_permutations, :]

    return selected_perms

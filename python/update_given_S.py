import numpy as np
import update_given_x as update

def update_given_S(mu, sigma, S, ind_ranking):
    """
    UPDATE_GIVEN_S Updates the mean and covariance given the word ranking.

    Args:
        mu (np.array): Prior mean vector.
        sigma (np.array): Prior covariance matrix.
        S (np.array): Embeddings of words in set (matrix).
        ind_ranking (list): Index of ranking of words given by human.

    Returns:
        mu (np.array): Updated mean vector.
        sigma (np.array): Updated covariance matrix.
    """

    # Convert ind_ranking to NumPy array (ensure integer type)
    ind_ranking = np.array(ind_ranking, dtype=int)

    while len(ind_ranking) >= 2:
        #print("Before update:")
        #print("mu:", mu)
        #print("sigma:\n", sigma)
        #print("S:\n", S)
        print("ind_ranking:", ind_ranking)

        # Call update_given_x function (ensure indexing matches MATLAB)
        mu, sigma = update.update_given_x(np.copy(mu), np.copy(sigma), np.copy(S), int(ind_ranking[0]))

        # print("\nAfter update_given_x:")
        # print("mu:", mu)
        # print("sigma:\n", sigma)

        # Remove column correctly from S
        original_index = ind_ranking[0]  # Store original index
        S = np.delete(S, original_index, axis=1)

        # Update ind_ranking *before* modifying it
        ind_ranking = np.delete(ind_ranking, 0)  # Remove first element

        # Adjust indices (decrement indices greater than removed element)
        ind_ranking = np.array([i - 1 if i > original_index else i for i in ind_ranking], dtype=int)

        print("\nAfter updating S and ind_ranking:")
        print("S:\n", S)
        print("ind_ranking:", ind_ranking)
        print("-" * 50)

    # Force symmetry if necessary
    if not np.allclose(sigma, sigma.T, atol=1e-8):  # Use tight tolerance
        print("Forcing symmetry")
        sigma = (sigma + sigma.T) / 2

    return mu, sigma

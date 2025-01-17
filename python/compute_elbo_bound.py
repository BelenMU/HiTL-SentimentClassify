import numpy as np

def compute_elbo_bound(mu_q, sigma_inv, S, sigma, prior_muxsigma, x, w):
    """
    Computes an upper bound on the Evidence Lower Bound (ELBO).
    
    Parameters:
    mu_q            - Mean of the variational distribution (vector).
    sigma_inv       - Inverse of the covariance matrix (matrix).
    S               - Embeddings of words in the set (matrix).
    sigma           - Covariance matrix (matrix).
    prior_muxsigma  - Prior mean multiplied by inverse of sigma (vector).
    x               - Selected word (vector).
    w               - Scalar weight.
    
    Returns:
    elbo            - Calculated upper bound on ELBO.
    """
    
    # Calculate the log-sum lower bound
    exp_term = (S.T @ mu_q) + 0.5 * np.sum((S.T @ sigma) * S.T, axis=1)
    log_sum = np.log(np.sum(np.exp(exp_term), axis = 0))

    # Compute the bound on ELBO
    elbo = - (x.T @ mu_q) + w * 0.5 * mu_q.T @ sigma_inv @ mu_q \
           - w * (prior_muxsigma @ mu_q) \
           + log_sum + 20 * (np.linalg.norm(mu_q) - 1) ** 2

    # Check for NaN or infinity values in the output
    if np.isnan(elbo):
        print("ELBO out NaN")
    elif np.isinf(elbo):
        print("ELBO out Inf")

    return elbo

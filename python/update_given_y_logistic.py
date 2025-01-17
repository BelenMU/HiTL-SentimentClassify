import numpy as np

def update_given_y_logistic(x, y, sigma, mu, tol=1e-5, max_iterations=1000):
    """
    Updates the posterior distribution of the parameters in a logistic regression model given the word's label.
    It follows the method by T. S. Jaakkola and M. I. Jordan,
    "Bayesian parameter estimation via variational methods," Statistics and Computing, vol. 10, pp. 25–37, 2000.
    
    Parameters:
    x      - Word embedding (numpy array).
    y      - Word label (integer, 1 or -1).
    sigma  - Prior covariance of the classifier (numpy array).
    mu     - Prior mean of the classifier (numpy array).
    tol    - Tolerance for convergence (default: 1e-5).
    max_iterations - Maximum number of iterations (default: 1000).
    
    Returns:
    mu_pos     - Posterior mean of the classifier (numpy array).
    sigma_pos  - Posterior covariance of the classifier (numpy array).
    """
    xi = np.sqrt(x @ sigma @ x.T + (x @ mu) ** 2)
    sigma_inv = np.linalg.inv(sigma)
    mu_prior = mu.copy() 
    
    difference = 1
    iteration_count = 0

    x = x.reshape(-1, 1)

    # Iterative update
    while difference > tol:
        # Compute posterior covariance
        sigma_pos = np.linalg.inv((sigma_inv + np.tanh(xi / 2) / (2 * xi) * (x @ x.T)))
        
        # Compute posterior mean
        mu = mu.reshape(-1, 1)
        mu_pos = sigma_pos @ (sigma_inv @ mu + (y / 2) * x)
        mu_pos = mu_pos / np.linalg.norm(mu_pos)
        mu_pos = mu_pos.flatten()
        # Update xi
        xi = np.sqrt(x.T @ sigma_pos @ x + (x.T @ mu_pos) ** 2)

        # Check for stopping condition
        difference = np.linalg.norm(mu_pos - mu_prior)
        mu_prior = mu_pos.copy()
        iteration_count += 1
        
        if iteration_count > max_iterations:
            break
            
    return mu_pos, sigma_pos

import numpy as np
from scipy.optimize import minimize
import compute_elbo_bound as compute
import fh
# Define the update_given_x function
def update_given_x(mu, sigma, S, ind_x):
    x = S[:, ind_x].reshape(-1, 1) # OK
    sigma_inv = np.linalg.inv(sigma) # OK, Has same values
    mu_prior = mu.copy().reshape(-1, 1) # OK, Has same values
    prior_muxsigma = (mu.T @ sigma_inv) # Has same values
    w = 1 # OK
    difference = 1 # OK
    iteration_count = 0 # OK

    while difference > 1e-4:
        # Update mean
        mu_update = lambda mu_prior: compute.compute_elbo_bound(mu_prior, sigma_inv, S, sigma, prior_muxsigma, x, w)
        options = {'maxiter': 1e5, 'disp': False}
        res_mu = minimize(mu_update, mu, jac = False, options=options)
        mu = (res_mu.x / np.linalg.norm(res_mu.x)) # Has same values

        # Update covariance
        L0 = np.linalg.cholesky(sigma)
        shape = L0.shape
        c_update = lambda L: fh.fh(L, sigma_inv, S, mu, w, shape) # OK
        res_L = minimize(c_update, L0.flatten(), method='L-BFGS-B', jac = True, options={'maxiter': int(1e5), 'disp': False})
        L = res_L.x.reshape(shape)

        if np.any(np.diag(L0) <= 0): # OK
            temp = np.where(np.diag(L0) <= 0)  # Find indices where diagonal is <= 0
            L[temp, temp] = 1e-14

        sigma = L @ L.T # OK

        difference = np.linalg.norm((mu - mu_prior), ord = 2) # OK
        mu_prior = mu.copy() # OK
        iteration_count += 1 # OK

        if iteration_count > 10: # OK
            break

    return mu, sigma

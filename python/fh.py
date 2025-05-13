import numpy as np

# If weight is not given, set it to 1
def fh(chol_matrix_flat, sigma_inv, S, mu, weight = 1, shape = (3, 3)):
    """
    Provides output and gradient of the objective function for covariance update 
    given the example selected by a human.
    
    Parameters:
    chol_matrix    - Cholesky decomposition of  the covariance matrix of 
                     the variational distribution (lower triangular matrix).
    sigma_inv      - Inverse of the prior covariance matrix.
    S              - Embeddings of words in the set.
    mu             - Mean of the variational distribution.
    weight         - Scale regularization to prior (default: 1).
    
    Returns:
    output         - Value of the objective function.
    gradient       - Gradient of the objective function at the given variational matrix.
    """
    chol_matrix = chol_matrix_flat.reshape(shape)
    # Ensuring that the Cholesky decomposition is a lower triangular matrix
    chol_matrix = np.tril(chol_matrix) # OK
    # Ensure no negative diagonal entries
    diag_values = np.diag(chol_matrix) # OK
    neg_diag_ind = np.where(diag_values <= 0)[0] # OK
    chol_matrix[neg_diag_ind, neg_diag_ind] = 10 ** (-14) # OK
    # Compute exp_term and log_sum
    exp_term = (S.T @ mu.T).reshape(-1, 1) + (0.5 * np.sum((S.T @ (chol_matrix @ chol_matrix.T)) * S.T, axis=1)).reshape(-1, 1) # OK
    log_sum = np.log(np.sum(np.exp(exp_term), axis=0)) # OK
    # Compute the final output
    output = -weight * np.sum(np.log(np.diag(chol_matrix))) + weight * 0.5 * np.trace(sigma_inv @ (chol_matrix @ chol_matrix.T)) + log_sum # OK
    # Compute the gradient (LLt and quadratic terms)
    LLt = chol_matrix @ chol_matrix.T # OK
    quadratic_terms = np.sum((S.T @ LLt) * S.T, axis=1).reshape(-1, 1)  # OK
    exp_terms = np.exp((S.T @ mu.T).reshape(-1, 1) + 0.5 * quadratic_terms) # OK
    exp_terms_flat = exp_terms.flatten() # OK
    # Compute the gradient
    gradient = weight * np.diag(1.0 / np.diag(chol_matrix)) \
           - ((weight * sigma_inv) + S @ np.diag(exp_terms_flat) @ S.T) @ chol_matrix
    gradient = -gradient # OK
    # Check for numerical stability: Adjust if gradient norm is too large
    if np.linalg.norm(gradient) > 1e30:
        gradient = gradient / 1e10
    # OK
    # Display warnings in case of NaN or Inf in gradient or output
    if np.isnan(gradient).any():
        print("Gradient has NaN values")
    elif np.isinf(gradient).any():
        print("Gradient has Inf values")
    
    if np.isnan(output):
        print("Objective function output has NaN values")
    elif np.isinf(output):
        print("Objective function output has Inf values")

    return output, gradient.flatten()
    #return output

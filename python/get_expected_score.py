import numpy as np

def get_expected_score(S, mu, sigma, noise_factor, scale, query_selected=1, num_particles=1e4):
    """
    Computes the terms of the active learning heuristic: 
    the difference between the entropy of the expected label and the expected 
    entropy of the label given the classifier.
    
    Parameters:
    S              - Embeddings of words in the set (matrix).
    mu             - Mean of the classifier distribution (vector).
    sigma          - Covariance matrix of the classifier distribution (matrix).
    noise_factor   - Scaling factor to control noise level model of the labeling.
    scale          - Scaling factor to control noise level in the model of the example selection.
    query_selected - 1: q_pos or 2: q_neg (default: 1).
    num_particles  - Number of particles used for the approximation (default: 1e4).
    
    Returns:
    EH - The expected entropy of the signal (E(H(Y))).
    HE - The entropy of the expected signal (H(Expectation(Y))).
    """
    
    # Define likelihood functions modeling the human response
    def p_y1_xandtheta(theta, S):
        return 1.0 / (1.0 + np.exp(- noise_factor * (theta.T @ S)))

    if query_selected == 1:
        softmax_num = lambda theta, S: np.exp(scale * (theta.T @ S))
    else:
        softmax_num = lambda theta, S: np.exp(-scale * (theta.T @ S))

    # Sample and normalize particles
    particles = np.random.multivariate_normal(mu, sigma, num_particles).T
    particles = particles / np.linalg.norm(particles, axis=0)  # Normalize each particle

    # Compute p(Y=1|x,theta) for all particles and embeddings
    p_y1_words = p_y1_xandtheta(particles, S)
    p_y_all = np.hstack([p_y1_words, 1 - p_y1_words])  # Combine for y=1 and y=-1

    # Compute joint discrete pmf f(x, y)
    softmax_num_words = softmax_num(particles, S)
    softmax_den_words = np.sum(softmax_num_words, axis=1)
    softmax_den_words = softmax_den_words.reshape(-1, 1) # OK until this point
    p_x_particles_all = softmax_num_words / softmax_den_words
    p_xy_particles_all = p_y_all * np.tile(p_x_particles_all, (1, 2))

    # Compute the entropy scores
    xylogxy_all = np.sum(p_xy_particles_all * np.log2(p_xy_particles_all), axis=1) 
    EH = np.mean(-xylogxy_all)

    E_p_xy = np.mean(p_xy_particles_all, axis=0)
    HE = -np.sum(E_p_xy * np.log2(E_p_xy))

    return EH, HE

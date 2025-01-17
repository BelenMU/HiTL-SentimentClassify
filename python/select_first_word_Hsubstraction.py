import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio



def select_first_word_Hsubstraction(list_embeddings, mu, sigma, noise_factor, num_particles=int(1e4)):
    """
    Selects the first word in the set based on an active learning heuristic that maximizes 
    the difference between the entropy of the expected label and the expected entropy of the label given the classifier.
    
    Parameters:
    list_embeddings - Matrix where columns represent word embeddings (numpy array).
    mu              - Mean of the classifier distribution (numpy array).
    sigma           - Covariance matrix of the classifier distribution (numpy array).
    noise_factor    - Scaling factor to control noise level in the labeling.
    num_particles   - Number of particles used for the approximation (default: 1e4).
    
    Returns:
    word            - The selected word embedding from the list (numpy array).
    index           - The index of the selected word in the list.
    list_embeddings - Updated list of embeddings with the selected word removed (numpy array).
    """
    mat_data = sio.loadmat(r'C:\Users\yslee\OneDrive\Desktop\Research\main\particles.mat')
    particles = mat_data['particles']
    # Define likelihood from the Bradley-Terry model
    def p_y1_xandtheta(theta, S):
        return 1.0 / (1.0 + np.exp(-(noise_factor * (theta.T @ S))))
    
    # Sample particles from theta's Gaussian distribution and normalize
    # particles = np.random.multivariate_normal(mu, sigma, num_particles).T
    particles = particles / np.linalg.norm(particles, axis=0)  # Normalize each particle (301, 10000)
    # Compute the heuristic: H(E(Y)) - E(H(Y))
    p_y1_particles_words = p_y1_xandtheta(particles, list_embeddings)
    p_y1_particles_words = np.maximum(p_y1_particles_words, 1e-12)  # Avoid log(0) by using a small value (10000, 353
    p_y1_particles_words = np.minimum(p_y1_particles_words, 1-1e-12)  # Avoid log(0) by using a small value 
    E_H_particle = np.mean(-p_y1_particles_words * np.log2(p_y1_particles_words) - 
                           (1 - p_y1_particles_words) * np.log2(1 - p_y1_particles_words), axis = 0).reshape(-1, 1).T # (1, 3532)
    E_p_y1 = np.mean(p_y1_particles_words, axis = 0).reshape(-1, 1).T # (1, 3532)
    H_E_particle = -E_p_y1 * np.log2(E_p_y1) - (1 - E_p_y1) * np.log2(1 - E_p_y1) # (1, 3532)
    H_ratio = H_E_particle - E_H_particle # (1, 3532)
    # Find the word corresponding to the maximal heuristic value
    index = np.argmax(H_ratio)
    # print("index: ")
    # print(index)
    word = list_embeddings[:, index]
    word = word.reshape(-1, 1)
    # Remove the selected word from the list of embeddings
    list_embeddings = np.delete(list_embeddings, index, axis=1)

    return word, index, list_embeddings

import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio
def select_next_word_Hsubstraction(previous_words, previous_index, list_embeddings, mu, sigma, noise_factor, scale, query=1, num_particles=int(1e4)):
    """
    Selects the next word in the set based on an active learning heuristic 
    that maximizes the difference between the entropy of the expected label 
    and the expected entropy of the label given the classifier.
    
    Parameters:
    previous_words  - Matrix where columns represent previously selected word embeddings (numpy array).
    previous_index  - List of indices of previously selected words (list).
    list_embeddings - Matrix where columns represent remaining word embeddings (numpy array).
    mu              - Mean of the classifier distribution (numpy array).
    sigma           - Covariance matrix of the classifier distribution (numpy array).
    noise_factor    - Scaling factor to control noise level in the labeling.
    scale           - Scaling factor to control noise level in the model of the example selection.
    query           - 1: q_pos or 2: q_neg (default: 1).
    num_particles   - Number of particles used for the approximation (default: 1e4).
    
    Returns:
    words           - Updated set of selected word embeddings including the new one (numpy array).
    index           - Updated list of indices of selected words (list).
    list_embeddings - Updated list of embeddings with the selected word removed (numpy array).
    max_score       - Score of the heuristic for the word set with the selected word.
    """
    mat_data = sio.loadmat(r'C:\Users\yslee\OneDrive\Desktop\Research\main\particles.mat')
    particles = mat_data['particles']
    # Define likelihood functions modeling the human response
    def p_y1_xandtheta(theta, S):
        return 1.0 / (1.0 + np.exp(-(noise_factor * (theta.T @ S))))

    if query == 1:
        softmax_num = lambda theta, S: np.exp((scale * (theta.T @ S)))
    else:
        softmax_num = lambda theta, S: np.exp(-(scale * (theta.T @ S)))

    # Sample and normalize particles
    # particles = np.random.multivariate_normal(mu, sigma, num_particles).T
    particles = particles / np.linalg.norm(particles, axis=0, ord = 2)  # Normalize each particle

    num_words = list_embeddings.shape[1]
    num_prev = previous_words.shape[1]
    # py1_all is a N_particles x (N_S*2 + 2) for previous words and new word options
    py1_all = np.zeros((num_particles, num_prev * 2 + 2, num_words))

    # Compute likelihoods for previous words and the new word
    p_y1_previous_words = p_y1_xandtheta(particles, previous_words)
    p_y1_particles_words = p_y1_xandtheta(particles, list_embeddings)

    # Replicate previous word likelihoods across all words
    p_y1_previous_words_rep = np.repeat(p_y1_previous_words[:, :, np.newaxis], num_words, axis=2)
    p_y1_previous_words_neg_rep = np.repeat(1 - p_y1_previous_words[:, :, np.newaxis], num_words, axis=2)
    # Sus but OK
    py1_all[:, 0:num_prev, :] = p_y1_previous_words_rep
    py1_all[:, num_prev, :] = p_y1_particles_words
    py1_all[:, num_prev + 1:-1, :] = p_y1_previous_words_neg_rep
    py1_all[:, -1, :] = 1 - p_y1_particles_words
    # Compute joint discrete pmf f(x, y)
    p_xy_particles_all = np.zeros((num_particles, num_prev + 1, num_words))
    
    # Softmax for previous and current words
    softmax_num_previous_words = softmax_num(particles, previous_words)
    softmax_num_words = softmax_num(particles, list_embeddings)
    softmax_num_words_sum = np.sum(softmax_num_previous_words, axis=1, keepdims = True) + softmax_num_words
    p_xy_particles_all[:, 0:num_prev, :] = np.tile(np.expand_dims(softmax_num_previous_words, axis=-1), (1, 1, num_words))
    # Assign softmax_num_words to the last column along the second axis
    p_xy_particles_all[:, -1, :] = softmax_num_words
    # Normalize by softmax_num_words_sum along the second axis
    # OK until this part
    softmax_num_words_sum_reshaped = np.reshape(softmax_num_words_sum, (num_particles, 1, num_words))
    p_xy_particles_all = p_xy_particles_all / np.tile(softmax_num_words_sum_reshaped, (1, num_prev + 1, 1))
    # Multiply element-wise by py1_all and repeat along the second axis
    p_xy_particles_all = py1_all * np.tile(p_xy_particles_all, (1, 2, 1))

    # Ensure no values are below 1e-12
    p_xy_particles_all = np.maximum(p_xy_particles_all, 1e-12)

    # Compute the heuristic: H(Expectation(Y)) - E(H(Y))
    xylogxy_all = np.squeeze(np.sum(p_xy_particles_all * np.log2(p_xy_particles_all), axis=1))
    E_H_particle = np.mean(-xylogxy_all, axis=0)
    E_p_xy = np.squeeze(np.mean(p_xy_particles_all, axis=0))
    H_E_particle = -np.sum(E_p_xy * np.log2(E_p_xy), axis=0)
    H_ratio = H_E_particle - E_H_particle

    # Find the word corresponding to the maximal heuristic value
    max_score = np.max(H_ratio)
    new_index = np.argmax(H_ratio)
    word = list_embeddings[:, new_index]

    # Update list embeddings: remove selected word
    list_embeddings = np.delete(list_embeddings, new_index, axis=1)

    # Update index and words
    new_index = new_index + np.sum(previous_index <= new_index)
    index = np.append(previous_index, new_index)
    word = word.reshape(-1, 1)
    words = np.hstack([previous_words, word])

    return words, index, list_embeddings, max_score

import numpy as np
from scipy.stats import multivariate_normal
import select_permutations as select
from math import factorial, log2
import scipy.io as sio
def select_next_word_ranking(previous_words, previous_index, list_embeddings, mu, sigma, noise_factor, scale, 
                             N_particles=None, max_permutations=500):
    N_prev = previous_words.shape[1]    

    if N_particles is None:
        N_particles = max(200, int(2e4 / factorial(N_prev + 1)))

    N_words = list_embeddings.shape[1]

    # Define probability functions
    def p_y1_xandtheta(theta, S):
        return 1 / (1 + np.exp(- (noise_factor * (theta.T @ S))))
    
    def softmax_num(theta, S):
        return np.exp(scale * (theta.T @ S))
    mat_data = sio.loadmat(r'C:\Users\yslee\OneDrive\Desktop\Research\main\test_results_matlab.mat')
    particles = mat_data['particles_mat']
    # Sample and normalize particles
    # particles = multivariate_normal.rvs(mean=mu.flatten(), cov=sigma, size=N_particles).T
    # particles /= np.linalg.norm(particles, axis=0, keepdims=True)

    # Compute p_y1
    p_y1_words = np.zeros((N_particles, N_prev + 1, N_words), dtype=np.float32)
    p_y1_words[:, :N_prev, :] = np.expand_dims(p_y1_xandtheta(particles, previous_words), axis=2)
    p_y1_words[:, N_prev, :] = p_y1_xandtheta(particles, list_embeddings)

    p_y1_all1 = np.prod(p_y1_words, axis=1)
    p_y1_allm1 = np.prod(1 - p_y1_words, axis=1)

    temp_options = N_prev + 2
    p_y = np.zeros((N_particles, temp_options, N_words), dtype=np.float32)
    p_y[:, 0, :] = p_y1_all1
    p_y[:, -1, :] = p_y1_allm1

    print("\n[DEBUG] p_y1_all1 (Shape:", p_y1_all1.shape, "):", p_y1_all1[:5, :5])  # Debug First 5 rows and columns
    print("[DEBUG] p_y1_allm1 (Shape:", p_y1_allm1.shape, "):", p_y1_allm1[:5, :5])

    # Compute softmax values
    softmax_num_previous_words = softmax_num(particles, previous_words)
    softmax_num_words = softmax_num(particles, list_embeddings)

    # Compute heuristic with numerical stability
    p_xy_safe = np.clip(p_y, 1e-12, 1)  # Prevent zero values
    xylogxy_all = np.sum(p_y * np.log2(p_xy_safe), axis=1)
    E_H_particle = -np.mean(xylogxy_all, axis=0)

    E_p_xy = np.mean(p_y, axis=0)  # Fix entropy normalization
    E_p_xy_safe = np.clip(E_p_xy, 1e-12, 1)  # Prevent log(0)
    H_E_particle = -np.sum(E_p_xy * np.log2(E_p_xy_safe), axis=0)

    H_ratio = H_E_particle - E_H_particle

    print("\n[DEBUG] H_ratio (Top 10 Values):", np.sort(H_ratio)[-5:])  # Print the 10 largest entropy values
    print("[DEBUG] Max H_ratio:", np.max(H_ratio))

    # Select word with maximum entropy heuristic
    max_score = float(np.max(H_ratio))
    sorted_indices = np.argsort(-H_ratio)  # Sort in descending order
    new_index = sorted_indices[0]  # Select highest-ranked word directly

    # Check that new_index does not exceed available words
    new_index = min(new_index, list_embeddings.shape[1])  
    new_index = new_index + 1

    print("\n[DEBUG] Sorted Indices (First 10):", sorted_indices[:10])
    print("[DEBUG] Selected new_index BEFORE fix:", new_index)

    print("[DEBUG] Selected new_index AFTER fix:", new_index)

    # Select the corresponding word
    word = list_embeddings[:, new_index - 1][:, np.newaxis]  # Correct word selection
    list_embeddings = np.delete(list_embeddings, new_index - 1, axis=1)  # Ensure correct removal

    # Update index
    index = np.concatenate((previous_index, [new_index]))  # Ensure MATLAB-like indexing

    words = np.concatenate((previous_words, word), axis=1)

    print("\n[DEBUG] Final Index:", index)
    print("[DEBUG] Final Max Score:", max_score)
    
    return words, index, list_embeddings, max_score

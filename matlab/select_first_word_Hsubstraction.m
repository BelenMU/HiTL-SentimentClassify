function [word, index, list_embeddings] = select_first_word_Hsubstraction(list_embeddings, mu, sigma, ...
    noise_factor, num_particles)
% SELECT_FIRST_WORD_HSUBSTRACTION Selects the first word in the set based on
% active learning heuristic that maximizes the difference between the entropy
% of the expected label and the expected entropy of the label given the
% classifier
% 
% Input arguments:
%  list_embeddings - Matrix where columns represent word embeddings
%  mu - Mean of the classifier distribution
%  sigma - Covariance matrix of the classifier distribution
%  noise_factor - Scaling factor to control noise level in the labeling
%  num_particles - Number of particles used for the approximation (optional)
%
% Returns:
%  word - The selected word embedding from the list
%  index - The index of the selected word in the list
%  list_embeddings - Updated list, with the selected word removed

    %  If num_particles is not given, set a default value
    if nargin < 5
        num_particles = 1e4;
    end

    % Define likelihood from the Bradley-Terry model
    p_y1_xandtheta = @(theta, S) 1 ./ (1 + exp(- noise_factor * (theta' * S))); 

    % Sample particles from theta's Gaussian distribution and normalize
    particles = mvnrnd(mu, sigma, num_particles)';
    particles = particles ./ vecnorm(particles);

    % Compute heuristic: H(E(Y)) - E(H(Y))
    p_y1_particles_words = p_y1_xandtheta(particles, list_embeddings);
    p_y1_particles_words = max(p_y1_particles_words, 1e-12); % Remove zero probabilities to avoid issues with log(0)
    E_H_particle = mean(-p_y1_particles_words .* log2(p_y1_particles_words) - (1-p_y1_particles_words).*log2(1-p_y1_particles_words));
    E_p_y1 = mean(p_y1_particles_words);
    H_E_particle = -E_p_y1 .* log2(E_p_y1) - (1-E_p_y1).*log2(1-E_p_y1);
    H_ratio = H_E_particle - E_H_particle;

    % Find the word corresponding to the maximal heuristic value
    [~, index] = max(H_ratio); % Only return argmax, not the value itself
    word = list_embeddings(:, index);

    % Return list of words without the selected oned (S can't contain the same word twice)
    list_embeddings(:, index) = [];
end
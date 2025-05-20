function [words, index, list_embeddings, max_score] = select_next_word_Hsubstraction(previous_words, ...
    previous_index, list_embeddings, mu, sigma, noise_factor, scale, query, num_particles)
% SELECT_NEXT_WORD_HSUBSTRACTION Selects the next word in the set based on
% active learning heuristic that maximizes the difference between the entropy
% of the expected label and the expected entropy of the label given the
% classifier
% 
% Input arguments:
%  list_embeddings - Matrix where columns represent word embeddings
%  mu - Mean of the classifier distribution
%  sigma - Covariance matrix of the classifier distribution
%  noise_factor - Scaling factor to control noise level model of the labeling
%  scale - Scaling factor to control noise level in the model of the example selection
%  query - 1: q_pos or 2: q_neg
%  num_particles - Number of particles used for the approximation (optional)
%
% Returns:
%  word - The selected word embedding from the list
%  index - The index of the selected word in the list
%  list_embeddings - Updated list, with the selected word removed
%  max_score - Score of heuristic for word set with selected word

    %  If num_particles is not given, set a default value
    if nargin < 9
        num_particles = 1e4;
    end
    % If query is not given, set q_pos as default
    if nargin < 8
        query = 1; 
    end

    % Define likelihood functions modeling the human response
    p_y1_xandtheta = @(theta, S) 1 ./ (1 + exp(- noise_factor * (theta' * S))); 
    if query == 1
        softmax_num = @(theta, S) exp((scale * (theta' * S)));
    else
        softmax_num = @(theta, S) exp(-(scale * (theta' * S)));
    end

    % Sample and normalize particles
    particles = mvnrnd(mu, sigma, num_particles)';
    particles = particles ./ vecnorm(particles);

    num_words = size(list_embeddings, 2);
    num_prev = size(previous_words, 2);
    
    % py1_all is a N_particles x N_S*2 (num previous words + 1)*2 x N_words
    % -> *2 for y=1 and y=-1 options -> N_S*2 total combination of x and y
    py1_all = zeros(num_particles, num_prev*2 + 2, num_words);
    p_y1_previous_words = p_y1_xandtheta(particles, previous_words);
    p_y1_particles_words = p_y1_xandtheta(particles, list_embeddings);
    p_y1_previous_words_rep = permute(repmat(p_y1_previous_words, [1, 1, num_words]), [1, 2, 3]);
    p_y1_previous_words_neg_rep = permute(repmat(1-p_y1_previous_words, [1, 1, num_words]), [1, 2, 3]);
    py1_all(:, 1:num_prev, :) = p_y1_previous_words_rep;
    py1_all(:, num_prev + 1 , :) = p_y1_particles_words;
    py1_all(:, (num_prev + 2):(end-1), :) = p_y1_previous_words_neg_rep;
    py1_all(:, end , :) = 1 - p_y1_particles_words;

    % Computejoint discrete pmf f(x, y)
    p_xy_particles_all = zeros(num_particles, num_prev + 1, num_words);
    softmax_num_previous_words = softmax_num(particles, previous_words);
    softmax_num_words = softmax_num(particles, list_embeddings);
    softmax_num_words_sum = sum(softmax_num_previous_words, 2) + softmax_num_words;
    p_xy_particles_all(:, 1:num_prev, :) = repmat(softmax_num_previous_words, [1, 1, num_words]);
    p_xy_particles_all(:, end, :) = softmax_num_words;
    p_xy_particles_all = p_xy_particles_all./ repmat(reshape(softmax_num_words_sum, [num_particles, 1, num_words]), [1, num_prev+1, 1]);
    p_xy_particles_all = py1_all .* repmat(p_xy_particles_all, [1, 2, 1]);
    p_xy_particles_all = max(p_xy_particles_all, 1e-12); % Remove zero probabilities to avoid issues with log(0)

    % Compute the heuristic H(Expectation(Y)) - E(H(Y))
    xylogxy_all = squeeze(sum(p_xy_particles_all .* log2(p_xy_particles_all), 2));
    E_H_particle = mean(-xylogxy_all);
    E_p_xy = squeeze(mean(p_xy_particles_all));
    H_E_particle = -sum(E_p_xy .* log2(E_p_xy), 1);
    H_ratio = H_E_particle - E_H_particle;

    % Find the word corresponding to the maximal heuristic value
    [max_score, new_index] = max(H_ratio);
    word = list_embeddings(:, new_index);   

    % Update outputs
    list_embeddings(:, new_index) = [];
    new_index = new_index + sum(previous_index <= new_index);
    index = [previous_index, new_index];
    words = [previous_words, word];
end
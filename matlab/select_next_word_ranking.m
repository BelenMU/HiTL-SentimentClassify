function [words, index, list_embeddings, max_score] = select_next_word_ranking(previous_words, ...
    previous_index, list_embeddings, mu, sigma, noise_factor, scale, N_particles, max_permuations)
%SELECT NEXT WORD RANKING Select greedily the next word to add to the set for query based on
%heuristic of entropy ratios: H(Expectation(Y)) / E(H(Y)) for query of word ranking.
    N_prev = size(previous_words, 2);
    if nargin < 8
        N_particles = max(200, int32(2e4 / factorial(N_prev + 1)));
    end
    if nargin < 9
        max_permuations = 500;
    end
    
    N_words = size(list_embeddings, 2);
    p_y1_xandtheta = @(theta, S) 1 ./ (1 + exp(- noise_factor * (theta' * S))); 
    softmax_num = @(theta, S) exp((scale * (theta' * S)));
    % Sample and normalize particles
    particles = single(mvnrnd(mu, sigma, N_particles)');
    particles = particles ./ vecnorm(particles);

    %% Py
    p_y1_words = zeros(N_particles, N_prev + 1, N_words, 'single');
    p_y1_previous_words = p_y1_xandtheta(particles, previous_words);
    p_y1_particles_words = p_y1_xandtheta(particles, list_embeddings);
    p_y1_previous_words_rep = permute(repmat(p_y1_previous_words, [1, 1, N_words]), [1, 2, 3]);
    p_y1_words(:, 1:N_prev, :) = p_y1_previous_words_rep;
    p_y1_words(:, N_prev + 1 , :) = p_y1_particles_words;
    p_y1_all1 = squeeze(prod(p_y1_words, 2));
    p_y1_allm1 = squeeze(prod(1 - p_y1_words, 2));
    clearvars p_y1_previous_words p_y1_previous_words_rep p_y1_particles_words
    
    temp_options = N_prev + 2;
    p_y = zeros(N_particles, temp_options, N_words, 'single');
    %{
    if max(max(p_y1_all1)) < 0.05 && mean(mean(p_y1_all1)) < 0.010
        p_y(:, 1, :) = [];
        temp_options = temp_options - 1;
    else
        p_y(:, 1, :) = p_y1_all1;
    end
    if max(max(p_y1_allm1)) < 0.05 && mean(mean(p_y1_allm1)) < 0.010
        p_y(:, end, :) = [];
        temp_options = temp_options - 1;
    else
        p_y(:, end, :) = p_y1_allm1;
    end
    %}
    p_y(:, 1, :) = p_y1_all1;
    p_y(:, end, :) = p_y1_allm1;
    clearvars p_y1_all1 p_y1_allm1 

    softmax_num_previous_words = softmax_num(particles, previous_words);
    softmax_num_words = softmax_num(particles, list_embeddings);
    [~, top_option] = sort([mean(softmax_num_previous_words, 1), mean(mean(softmax_num_words))], 2, 'descend');
    ranking_options = select_permutations(N_prev, top_option, max_permuations);% 1e3);
    N_rankings = size(ranking_options, 1);
    %{
    if N_prev >=1
        disp(['N_prev = ' , num2str(N_prev) ,', N_particles = ' ,...
            num2str(N_particles) , ' and N_rankings ' ,num2str(N_rankings)])
    end
    %}
    p_y = repmat(p_y, [1, N_rankings, 1]);  
    clearvars top_options permutations

    for last_pos = 1:N_prev
        pos_ind = ranking_options(:, 1:last_pos);
        neg_ind = ranking_options(:, last_pos+1:end);
        for iter = 1:N_rankings
            p_y1_last_pos = prod(p_y1_words(:, pos_ind(iter, :), :), 2) .*...
                            prod(1 - p_y1_words(:, neg_ind(iter, :), :), 2); % Size N_particles, 1, N_words
            p_y(:, (iter-1)*temp_options + last_pos + 1, :) = p_y1_last_pos; % Size N_particles, N_rankings, N_words
        end
    end
    clearvars p_y1_last_pos p_y1_words
    
    %% P_x
    p_x_particles_first = zeros(N_particles, N_prev + 1, N_words, 'single');
    p_x_particles_first(:, 1:N_prev, :) = repmat(softmax_num_previous_words, [1, 1, N_words]);
    p_x_particles_first(:, end, :) = softmax_num_words;
    p_x_rankings = zeros(N_particles, N_rankings, N_words, 'single');
    clearvars softmax_num_previous_words softmax_num_words 

    for rr = 1:N_rankings
        current_ranking = ranking_options(rr, :);
        p_rr = ones(N_particles, 1, N_words, 'single');
        while length(current_ranking) > 1
            p_rr = p_rr .* ...
                p_x_particles_first(:, current_ranking(1), :) ./ ...
                sum(p_x_particles_first(:, current_ranking, :), 2);
            current_ranking = current_ranking(2:end);
        end
        p_x_rankings(:, rr, :) = p_rr;
    end
    clearvars current_ranking p_rr p_x_particles_first

    %% P_xy
    cols_to_repeat = kron(1:N_rankings, ones(1, temp_options));
    p_xy = p_x_rankings(:, cols_to_repeat, :) .* p_y;
    clearvars p_x_rankings p_y
    % Normalize
    p_xy = p_xy ./ (repmat(sum(p_xy, 2), [1, size(p_xy, 2), 1]));
    
    %% Heuristic
    % Compute the Entropy Ratios
    p_xy(p_xy<1e-10) = 1e-10;
    xylogxy_all = squeeze(sum(p_xy .* log2(p_xy), 2));
    E_H_particle = mean(-xylogxy_all);
    E_p_xy = squeeze(mean(p_xy)); 
    clearvars p_xy xylogxy_all
    E_p_xy(E_p_xy<1e-10) = 1e-10;
    H_E_particle = -sum(E_p_xy .* log2(E_p_xy), 1) ;
    H_ratio = H_E_particle - E_H_particle; % For N_prev 1 and 2 it is around 2.3, 2.0968, but for 3 it is 4.1 and 3.8
    [max_score, new_index] = max(H_ratio);
    word = list_embeddings(:, new_index);    
    list_embeddings(:, new_index) = [];
    new_index = new_index + sum(previous_index <= new_index);
    index = [previous_index, new_index];
    words = [previous_words, word];
end
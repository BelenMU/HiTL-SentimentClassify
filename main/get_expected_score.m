function [EH, HE] = get_expected_score(S, mu, sigma, ...
    noise_factor, scale, query_selected, num_particles)
% GET_EXPECTED_SCORE Computes the terms of the active learning heuristic:
% the difference between the entropy of the expected label and the expected 
% entropy of the label given the classifier
% 
% Input arguments:
%  S                - Embeddings of words in set.
%  mu               - Mean of the classifier distribution
%  sigma            - Covariance matrix of the classifier distribution
%  noise_factor     - Scaling factor to control noise level model of the labeling
%  scale            - Scaling factor to control noise level in the model of the example selection
%  query_selected   - 1: q_pos or 2: q_neg
%  num_particles    - Number of particles used for the approximation (optional)
%
% Returns:
%  EH - The expected entropy of the signal (E(H(Y)))
%  HE - The entropy of the expected signal (H(Expectation(Y)))

    %  If num_particles is not given, set a default value
    if nargin < 9
        num_particles = 1e4;
    end
    %  If query is not given, set q_pos
    if nargin < 8
        query_selected = 1;
    end

    % Define likelihood functions modeling the human response
    p_y1_xandtheta = @(theta, S) 1 ./ (1 + exp(- noise_factor * (theta' * S))); 
    if query_selected == 1
        softmax_num = @(theta, S) exp((scale * (theta' * S)));
    else
        softmax_num = @(theta, S) exp(-(scale * (theta' * S)));
    end

    % Sample and normalize particles
    particles = mvnrnd(mu, sigma, num_particles)';
    particles = particles ./ vecnorm(particles);

    % Compute H(Expectation(Y)) and E(H(Y))
    % py1_all is a N_particles x N_S*2
    % -> *2 for y=1 and y=-1 options -> N_S*2 total combination of x and y
    p_y1_words = p_y1_xandtheta(particles, S);
    p_y_all = [p_y1_words, 1-p_y1_words];

    % Computejoint discrete pmf f(x, y)
    softmax_num_words = softmax_num(particles, S);
    softmax_den_words = sum(softmax_num_words, 2);
    p_x_particles_all = softmax_num_words ./ softmax_den_words;
    p_xy_particles_all= p_y_all .* repmat(p_x_particles_all, [1, 2]);

    % Compute the Entropy Scores
    xylogxy_all = sum(p_xy_particles_all .* log2(p_xy_particles_all), 2);
    EH = mean(-xylogxy_all);
    E_p_xy = mean(p_xy_particles_all);
    HE = -sum(E_p_xy .* log2(E_p_xy));
end
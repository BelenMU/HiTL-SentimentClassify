function [mu, sigma] = update_given_S(mu,sigma, S, ind_ranking)
% UPDATE_GIVEN_S Updates the mean and covariance given the word ranking.

% We assume (Plackett-Luce model) that the probability of a ranking outcome
% is the product of the probabilities of choosing an item among the set of
% remaining items at each step, and removing top word one at a time.
%
% Input arguments:
%   mu     - Prior mean vector.
%   sigma  - Prior covariance matrix.
%   S      - Embeddings of words in set.
%   ind_ranking  - Index of raking of words given by human
%
% Returns:
%   mu     - Updated mean vector.
%   sigma  - Updated covariance matrix.

    while length(ind_ranking)>=2
        % Calling the function with the current index
        [mu, sigma] = update_given_x(mu, sigma, S,  ind_ranking(1));
        
        % Deleting the ind_ranking(1)-th column from S
        S(:, ind_ranking(1)) = [];
        
        % Updating the ind_ranking vector
        % Since we remove a column from S, we need to decrement the indices that are
        % greater than the one we just removed.
        ind_ranking = ind_ranking - (ind_ranking > ind_ranking(1));
        
        % Now we also need to remove the first element since we just processed it
        ind_ranking(1) = [];
    end
    if (~issymmetric(sigma))
        disp("Forcing symmetry")
        sigma = (sigma + sigma.')/2;
    end
end
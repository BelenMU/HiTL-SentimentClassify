function [mu, sigma] = update_given_x(mu,sigma, S, ind_x)
% UPDATE_GIVEN_X Updates the mean and covariance given the selected word.
%
% Input arguments:
%   mu     - Prior mean vector.
%   sigma  - Prior covariance matrix.
%   S      - Embeddings of words in set.
%   ind_x  - Index of word in set selected by human.
%
% Returns:
%   mu     - Updated mean vector.
%   sigma  - Updated covariance matrix.

    x = S(:, ind_x);
    sigma = double(sigma);
    sigma_inv = inv(sigma);
    mu = double(mu);
    mu_prior = mu;

    % Initialize
    prior_muxsigma = mu' * sigma_inv;
    w = 1;
    difference = 1;
    iteration_count = 0;
    
    while(difference > 1e-4)   
        % Update mean
        mu_update = @(mu_prior) compute_elbo_bound(mu_prior, sigma_inv, S, sigma, prior_muxsigma, x, w);
        options = optimoptions(@fminunc, 'SpecifyObjectiveGradient',false,'MaxIterations', 1e5, 'Display', 'off'); %'HessianFcn' , 'objective' , 
        mu = fminunc(mu_update, mu, options);
        mu = mu./norm(mu);

        % Update covariance
        c_update = @(L) fh(L, sigma_inv, S, mu, w);
        options = optimoptions(@fminunc, 'HessianApproximation' , 'lbfgs' , 'SpecifyObjectiveGradient', true,'MaxIterations', 1e5, 'Display', 'off'); % 'FiniteDifferenceStepSize', 1e-5,
        L0 = chol(sigma, 'lower');
        L = fminunc(c_update, L0, options);
        if(diag(L0)<=0)
            temp = find(diag(L0)<=0);
            L(temp, temp) = 1e-14;
        end
        sigma = L*L';

        % Check termination conditions
        difference = norm(mu-mu_prior);
        mu_prior = mu;
        iteration_count = iteration_count + 1;
        if (iteration_count>10)
            break
        end
    end
end
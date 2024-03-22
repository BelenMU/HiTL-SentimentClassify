function [mu_pos, sigma_pos] = update_given_y_logistic(x, y, sigma, mu)
%UPDATE_GIVEN_Y_LOGISTIC Updates the posterior distribution of the parameters
% in a logistic regression model given the word's label. 
% It follows the method byT. S. Jaakkola and M. I. Jordan,
% "Bayesian parameter estimation via variational methods," 
% Statistics and Computing, vol. 10, pp. 25–37,2000.
%
% Input arguments:
%   x      - Word embedding.
%   y      - Word label.
%   sigma  - Prior covariance of the classifier.
%   mu     - Prior mean of the classifier.
%
% Returns:
%   mu_pos     - Posterior mean of the classifier.
%   sigma_pos  - Posterior covariance of the classifier.

    xi = sqrt(x'*sigma*x + (x'*mu)^2);
    sigma_inv = inv(sigma);
    mu_prior = mu;
    difference = 1;
    iteration_count = 0;

    while(difference > 1e-5)
        % Compute posterior
        sigma_pos = inv(sigma_inv + tanh(xi/2) / (2*xi) * x*x');
        mu_pos = sigma_pos * (sigma_inv * mu + (y/2)*x);
        mu_pos = mu_pos ./ norm(mu_pos);
        xi = sqrt(x'* sigma_pos * x + (x'*mu_pos)^2);

        % Check for stopping condition
        difference = norm(mu_pos-mu_prior);
        mu_prior = mu_pos;
        iteration_count = iteration_count + 1;
        if (iteration_count>1000)
            break
        end
    end
end

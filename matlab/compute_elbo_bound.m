function [elbo] =  compute_elbo_bound(mu_q, sigma_inv, S, sigma, prior_muxsigma, x, w)
%MU_UPDATE_GIVEN_X computes an upperbound on the Evidence Lower BOund (ELBO).
%
% Input arguments:
% mu_q            - Mean of variational distribution.
% sigma_inv       - Inverse of the covariance matrix.
% S               - Embeddings of words in set.
% sigma           - Covariance matrix.
% prior_mu_sigma  - Prior mean multiplied by inverse of sigma.
% x               - Selected word.
% w               - Scalar weight.
%
% Returns:
% elbo            - Calculated upper bound on ELBO.

    % Calculate the log-sum lower bound
    exp_term = bsxfun(@plus, S' * mu_q, 0.5 * sum((S' * sigma) .* S', 2));
    log_sum = log(sum(exp(exp_term), 1));

    % Compute the bound on ELBO
    elbo = - x'*mu_q + w * 0.5 * mu_q'*sigma_inv*mu_q ... % KL Gaussians
            - w * prior_muxsigma*mu_q...
            + log_sum + 20*(norm(mu_q)-1)^2;

    % Check for NaN or infinity values in the output.
    if isnan(elbo)
        disp("ELBO out NaN")
    elseif isinf(elbo)
        disp("ELBO out Inf")
    end
    elbo = double(elbo);
end
function [output, gradient] =  fh(chol_matrix, sigma_inv, S, mu, weight)
% FH provides output and gradient of objective function for covariance
% update given the example selected by human. It is important to provide
% the covariance to accelerate computation and not to overflow the memory.
%
% Input arguments:
%   chol_matrix     - Cholesky decomposition of covariance matrix of
%                     variational distribution
%   sigma_inv       - Inverse of prior covariance matrix.
%   S               - Embeddings of words in set.
%   mu              - Mean of variational distribution.
%   weight          - Scale regularization to prior.
%
% Returns:
%   output     - Value of objective function for given variational matrix.
%   gradient   - Gradient of objective function at given variational matrix.

    %  If weight is not given, set to 1
    if nargin < 5
        weight = 1;
    end

    % Ensuring that the Cholesky decomposition, cholMatrix, is 
    % a lower triangular matrix with real and positive diagonal entries.
    chol_matrix = tril(chol_matrix);
    neg_daig_ind = find(diag(chol_matrix)<=0);
    chol_matrix(neg_daig_ind, neg_daig_ind) = 10^(-14); 
    
    % Compute the output
    exp_term = bsxfun(@plus, S' * mu, 0.5 * sum((S' * (chol_matrix*chol_matrix')) .* S', 2));
    log_sum = log(sum(exp(exp_term), 1));
    output = -weight*sum(log(diag(chol_matrix))) + weight*0.5 * trace(sigma_inv*(chol_matrix*chol_matrix'))...
        + log_sum;
    
    % Compute gradient
    LLt = chol_matrix * chol_matrix';
    quadratic_terms = sum((S' * LLt) .* S', 2); % S' * (L*L') * S for each column
    exp_terms = exp(S' * mu + 0.5 * quadratic_terms);

    gradient = weight * diag(1 ./ diag(chol_matrix)) ...
               - ((weight * sigma_inv) ...
               + S * diag(exp_terms) * S') * chol_matrix;
    gradient = -gradient;    
 
    % Check for numerical stability: Adjust if gradient norm is too large
    if norm(gradient)> 1e30
        gradient = gradient ./ 1e10;
    end
 
    % Display warnings in case of NaN or Inf in gradient or output
    if isnan(gradient)
        disp("Gradient has NaN values");
    elseif isinf(gradient)
        disp("Gradient has Inf values");
    end
    if isnan(output)
        disp("Objective function output has NaN values");
    elseif isinf(output)
        disp("Objective function output has Inf values");
    end
end
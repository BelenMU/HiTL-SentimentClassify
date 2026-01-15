function plot_bin_mean_variance(width, prod_min, prod_max, prod, data, estimation, y_label_text, legend_text, fit_sigmoid)
% PLOT_BIN_MEAN_VARIANCE Plots the mean and variance of data in bins based
% n
% on the provided range, overlaying a sigmoid fit or estimation plot.
%
% This function divides the range of inner product values into bins and 
% calculates the mean and variance of the associated data for each bin. 
% It then plots these statistics using error bars to represent the variance.
% As well as the estimated values by linear fit or Sigmoid approximation.
%
% Input arguments:
%  width           - The width of each bin for dividing the range of inner product values.
%  prod_min        - The minimum inner product value considered for binning.
%  prod_max        - The maximum inner product value considered for binning.
%  prod            - Vector containing the inner products corresponding to 'data'.
%  data            - Vector containing the dependent variable values to be binned and analyzed.
%  estimation      - Array containing the estimated values from the linear fit for overlaying purposes.
%  y_label_text    - Text label for the y-axis of the plot (accepts LaTeX formatted strings).
%  legend_text     - Text for the legend entry corresponding to the error bars plot (accepts LaTeX formatted strings).
%  fit_sigmoid     - Optional parameter; if not false, specifies parameters 
%                    for fitting a sigmoid function on the binned data.
%
% Returns:
%   - This function does not return any values but generates a plot.

    if nargin < 9
        fit_sigmoid = false; % Do not try to fit with Sigmoid
    end
    
    % Initialization
    n_bins = (prod_max - prod_min) / width + 2;
    prod_bins = zeros(1, n_bins);
    score_bins = zeros(1, n_bins);
    scoreVar_bins = zeros(1, n_bins);

    % Handle all outliers below prod_min as one bin
    temp = find(prod < prod_min);
    prod_bins(1) = mean(prod(temp));
    score_bins(1) = mean(data(temp));
    scoreVar_bins(1) = var(data(temp));

    % Compute the mean and variance per bin
    for ii = 1:n_bins-1    
        temp = find(prod > prod_min+width*(ii-1) & prod < prod_min+width*ii);
        prod_bins(ii+1) = mean(prod(temp));
        score_bins(ii+1) = mean(data(temp));
        scoreVar_bins(ii+1) = var(data(temp));
    end

    % Handle all outliers above prod_max as one bin
    temp = find(prod > prod_max);
    prod_bins(end) = mean(prod(temp));
    score_bins(end) = mean(data(temp));
    scoreVar_bins(end) = var(data(temp));
    
    % Plot Figure
    figure; errorbar(prod_bins, score_bins, sqrt(scoreVar_bins),'-s',...
        'MarkerEdgeColor','red','MarkerFaceColor','red')
    hold on;

    if fit_sigmoid ~= false % Include linear estimation for score
        y = 1 ./ (1 + exp(-fit_sigmoid*prod_bins));
        plot(prod_bins, y, 'g-', 'linewidth', 2)
        xlim([-0.25, 0.25])
        ylim([-0.2, 1.2])
    else % Include Sigmoid estimation for label
        plot(prod, estimation, 'g-', 'linewidth', 2);
        xlim([-0.2, 0.2])
        ylim([-2.5, 2.5])
    end
    xlabel("$\mathbf{x^T\theta}$", 'interpreter', 'latex')
    ylabel(y_label_text, 'Interpreter','latex')
    grid on
    legend( legend_text, 'interpreter', 'latex', 'location', 'nw', 'fontsize', 12)
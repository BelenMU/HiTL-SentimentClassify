function plot_with_errorbars(data_matrix, type, color)
% PLOT_WITH_ERRORBARS  Plots data with error bars.
% This function plots the mean of given data along with error bars representing
% the standard error of the mean. It supports plotting Mean Square Error (MSE)
% or accuracy. MSE plots are displayed on a logarithmic scale.
% 
% Input arguments:
%  data_matrix      - A matrix where each column represents a set of observations
%                     for a given iteration, and each row corresponds to a 
%                     different initialization.
%  type             - Character specifying the data type for plotting. 
%                     'm' for Mean Square Error (MSE) or 'a' for accuracy. 
%                     Optional; defaults to 'm'.
%  color            - Character or string specifying the line color for the 
%                     plot. Optional; defaults to 'b' (blue).
% Returns:
%   - This function does not return any values but generates a plot.

    if nargin < 2
        type = 'm'; % MSE
    end
    if nargin < 3
        color = 'b'; % blue
    end
    if type == 'a'
        data_matrix = data_matrix .* 100;
    end

    % Calculate means and standard errors
    data_mean = mean(data_matrix, 2);
    data_std = std(data_matrix, 0, 2) / sqrt(size(data_matrix, 2));
    num_iterations = size(data_matrix, 1) - 1;

    % Create figure
    figure;    
    % Plot the mean lines
    plot(0:num_iterations, data_mean, color, 'LineWidth', 2);
    hold on;
    % Add shaded error bars
    x_fill = [0:num_iterations, num_iterations:-1:0];
    fill_data = [data_mean'+data_std', fliplr(data_mean'-data_std')];
    fill(x_fill, fill_data, color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    grid on
    xlabel("$i$", 'interpreter', 'latex', 'fontsize', 13)

    if type == 'a'
        ylabel("Accuracy (%)", 'fontsize', 13)
        legend('Accuracy', 'interpreter', 'latex', 'fontsize', 12, 'location', 'se')
    else
        set(gca, 'YScale', 'log')
        ylabel("$\|\mathbf{\theta} - \mathrm{E}[\hat{\mathbf{\theta}}_i]\|_2^2$", ...
            'interpreter', 'latex', 'fontsize', 13)
        legend('MSE', 'interpreter', 'latex', 'fontsize', 12)
    end
    

end
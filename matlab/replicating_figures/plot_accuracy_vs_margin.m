function plot_accuracy_vs_margin(estimated_theta)% Input arguments:
% PLOT_ACCURACY_VS_MARGIN It plots the accuracy for different dictionary
% sizes according to the neutrality of the word, vs. the accuracy of the
% MMSE estimator
%
% Input arguments:
%  estimated_theta  - A matrix containing the estimated theta parameters.
%                     Each column represents for different time.
% Returns:
%   - This function does not return any values but generates a plot.
%
% Note: This code requires 'init_VI.mat' to be in the workspace directory, 

    load('init_VI.mat', 'list_score', 'list_var', 'theta','x_train_init','y_train_init')
    
    % Probability of a word being positive according to dataset
    Py1 = 1 - normcdf(0,list_score,sqrt(list_var));
    
    % Distance from complete neutrality in which to measure
    margins = [1e-5:0.01:0.49, 0.49];
    lm = length(margins);
    
    %% Ground Truth Accuracy
    accu_true_theta = zeros(1, lm);
    for ii = 1:lm    
        ind = find(Py1>=0.5-margins(ii) & Py1<=0.5+margins(ii));
        xtrain = x_train_init;
        y_train = y_train_init;
        xtrain(:, ind) = [];
        y_train(ind) = [];
    
        accu_true_theta(ii) =  1 - sum(abs(sign(theta' * xtrain) - y_train)) / (2*length(y_train));
    end
    
    %% Accuracy for estimated theta
    accu_estimated_theta = zeros(1, lm);
    for ii = 1:lm
        xtrain = x_train_init;
        y_train = y_train_init;
        
        ind = find(Py1>=0.5-margins(ii) & Py1<=0.5+margins(ii));
        xtrain(:, ind) = [];
        y_train(ind) = [];
    
        for jj = 1:N_initializations %Initializations
            accu_estimated_theta(ii) =  accu_estimated_theta(ii) + 1 - ...
                sum(abs(sign(estimated_theta(:, jj)' * xtrain) - y_train)) / (2*length(y_train));
        end
    end
    accu_estimated_theta = accu_estimated_theta ./ N_initializations;
    
    %% Plot Gap vs. Dictionary entries
    figure; scatter(margins, accu_true_theta.*100, 'o','linewidth', 2)
    hold on; scatter(margins, accu_estimated_theta.*100, 'o','linewidth', 2)
    grid on
    xlabel("$\delta$", 'interpreter', 'latex', 'fontsize', 13)
    ylabel("Accuracy for entries with $|P_y -0.5|\geq \delta$", 'interpreter', 'latex', 'fontsize', 13)
    legend("Ground Truth", "Learned Classifier", 'fontsize', 12, 'location', 'se')
end
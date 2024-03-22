%% Enhancing Human-in-the-Loop Learning for Binary Sentiment Word Classification
% Belén Martín Urcelay, Christopher J. Rozell, Matthieu R. Bloch
% 03/2024
% Analysis of relationship between a word's scor and its distance to classifier
% It replicates the plots in Figure 1
%% Load dataset to analyze
load('list_freq2000.mat') % loads the most frequent words in 2000s dataset from SocialSent.
% load('list_adj2000.mat') % uncomment to analyze the adjective dataset in SocialSent instead.

%% Analysis of word's score vs. the inner product between the classifier and word
prod = theta' * list_embeddings;
fit_linear = polyfit(prod,list_score,1);
estimated_score = fit_linear(1) .* prod + fit_linear(2);

% Plot estimated vs. actual score
width = 0.005;
prod_min = -0.15;
prod_max = 0.15;
legend_text = ["Score", [num2str(fit_linear(1), 3), '$(x^T\theta)$', num2str(fit_linear(2), 2)]];
y_label_text = "Valence Score";
plot_bin_mean_variance(width, prod_min, prod_max, prod, list_score, ...
    estimated_score, y_label_text, legend_text)

%% Find Fit of error as a Gumbel distribution following Assumption II.1
pd = fitdist((list_score - estimated_score'), 'ev');
% pd(2) is equivalent to \sigma in eq. (3) and (4), necessary to compute 
% the word selection likelihood.

%% Analysis of label vs. the inner product between the classifier and word
% Number of times each word's score is sampled
num_rep = 5000;
% Sample score according to the dataset
label = repmat(list_score, num_rep, 1) + randn(num_words*num_rep, 1) .* sqrt(repmat(list_var, num_rep,1));

% Compute label -> positive scores y=1, and negative scores y=0
label(label>0) = 1;
a = find(label==0);
label(label<0) = 0;
label(a) = (sign(rand(size(a)) - 0.5)+1)./2;

% Plot label probability vs. the inner product between the classifier and word 
% and fit with Sigmoid function
width = 0.01;
prod_min = -0.2;
prod_max = 0.2;
prod_all_samples = repmat(prod, 1, num_rep);
scale_w = 13;   % Temperature of Sigmoid, fitted heuritically
legend_text = ["Probability", ['Sigmoid(', num2str(scale_w), '$x^T\theta)$']];
y_label_text = "$\mathbf{P}[y = 1]$";
plot_bin_mean_variance(width, prod_min, prod_max, prod_all_samples, label, ...
    "", y_label_text, legend_text, scale_w )
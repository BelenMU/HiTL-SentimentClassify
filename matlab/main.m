%% Enhancing Human-in-the-Loop Learning for Binary Sentiment Word Classification
% Belén Martín Urcelay, Christopher J. Rozell, Matthieu R. Bloch
% 03/2024

%% Parameters to set
filename = "results_N6.mat";     % Directory in which to save results 
num_iterations = 1000;             % Number of interactions with humans 
num_initializations = 10;         % Number of experiments
num_S = 6;                       % Number of words in query, $|\mathcal{S}|$
word_selection = true;            % Wether to include word selection in query
active = true;                    % Wether to include active word set selection
create_CSV = true;               % Wether to create CSV file with words from each iteration

%% Initialize
load("init_SocialSent_freq.mat"); % Load dataset from SocialSent and its parameters 

end_mu = zeros(d+1, num_initializations);
end_sigma = zeros(d+1, d+1, num_initializations);
mu_save = zeros(num_initializations, d+1, num_iterations);

C_root_det = zeros(num_iterations+1, num_initializations);
MSE_MT_init = zeros(num_iterations+1, num_initializations);
accu_init = zeros(num_iterations+1, num_initializations);
max_score_init = zeros(num_iterations, num_initializations);
indices_vector = zeros(num_S, num_iterations, num_initializations); %init vector of indices from each iteration

if word_selection == false
    num_S = 1;
    active = false;
end

%% Run algorithm
% Set up a parallel pool
if isempty(gcp('nocreate'))
    parpool; 
end

parfor init = 1:num_initializations
    disp("INITIALIZATION")
    mu = mvnrnd(mu_init, C_init)';         
    % To exactly replicate the figures with same initialization as the paper:  mu_init_experiments_paper(:, init);
    sigma = C_init;
    C_root_det_temp = zeros(num_iterations+1, 1);
    MSE_MT_init_temp = zeros(num_iterations+1, 1);
    accu_init_temp = zeros(num_iterations+1, 1);
    C_root_det_temp(1) = prod(nthroot(svd(sigma), (d+1)));
    MSE_MT_init_temp(1) = vecnorm(theta - mu)^2;
    accu_init_temp(1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / (2*length(y_train));
    max_score = 0;
    try
        for ii = 1:num_iterations
            mu_save(init, :, ii) = mu;
            % Select Query at random. 1: q_pos, 2: q_neg
            query_selected = randi([1,2]);

            % Select set of words to show to the teacher
            if active
                [S, ind_S, list_embeddings_remaining] = select_first_word_Hsubstraction(list_embeddings, mu, sigma, noise_factor);
                           

             for loop = 2:num_S
                    [S, ind_S, list_embeddings_remaining, max_score] = select_next_word_Hsubstraction(S, ...
                                                                ind_S, list_embeddings_remaining, mu, sigma, ...
                                                                noise_factor, scale, query_selected);    
                                    
             end
               indices_vector(:, ii, init) = ind_S(:);
                max_score_init(ii, init) = max_score;
            else
                ind_S = randperm(num_words, num_S);
                S = list_embeddings(:, ind_S);
                [EH, HE] = get_expected_score(S,mu, sigma, ...
                                              noise_factor, scale, query_selected);
                max_score_init(ii, init) = HE - EH;
            end

            % Model Human
            % Sample Score a human would give to each of the words
            sample_score = list_score(ind_S) + randn(num_S, 1) .* sqrt(list_var(ind_S));
            % Select word according to query
            if query_selected==1
                [score_x, ind_x] = max(sample_score);
            else
                [score_x, ind_x] = min(sample_score);
            end
            % Label word
            if score_x > 0
                y =1;
            else
                y = -1;
            end

            % Update given Label        
            x_t = S(:, ind_x);
            [mu, sigma] = update_given_y_logistic(x_t.*noise_factor, y, sigma, mu);
            
            % Update given example selected        
            if word_selection
                if query_selected==1
                    [mu, sigma] = update_given_x(mu, sigma, scale.*S, ind_x);
                else
                    [mu, sigma] = update_given_x(mu, sigma, -1*scale.*S, ind_x);
                end
            end
            C_root_det_temp(ii+1) = prod(nthroot(svd(sigma), (d+1)));
            MSE_MT_init_temp(ii+1) = vecnorm(theta - mu)^2;
            accu_init_temp(ii+1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / (2*length(y_train));

            % Display iteration number to track progress
            if mod(ii, 200)==0
                disp(['Iteration ', num2str(ii), '/', num2str(num_iterations), ' in initialization ', num2str(init)])
            end
        end
    catch ME
        fprintf('Error in worker %d: %s\n', spmdIndex, ME.message);
        fprintf('Iteration %d', ii)
    end




%{
    if create_CSV
	word_results = {num_iterations*num_initializations, num_S};
	for i=1:size(indices_vector,1)
	    for j=1:size(indices_vector,2)
		for k=1:size(indices_vector,3)
		    word_results{((k-1)*num_iterations)+j,1}=strcat('b',num2str(num_S),'_e_',num2str((k-1)*num_iterations+j));
		    index = indices_vector(i,j,k);
		    word_results{((k-1)*num_iterations)+j,i+1} = list_word{index};
		end
	    end
	end
	results_table = cell2table(word_results);
	writetable(results_table, strcat('output_words_N',num2str(num_S),'.csv'));
    end
%}




    % Save results from all initializations and threads
    C_root_det(:, init) = C_root_det_temp;
    MSE_MT_init(:, init) = MSE_MT_init_temp;
    accu_init(:, init) = accu_init_temp;
    end_mu(:, init) = mu;
    end_sigma(:, :, init) = sigma;
end
to_save = {'end_mu', 'end_sigma', 'mu_save', 'C_root_det', ...
    'MSE_MT_init', 'accu_init', 'max_score_init', 'indices_vector'};
save(filename, to_save{:})


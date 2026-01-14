function selection_init(filename, init, N_S, params)
%RANKING INIT Runs human in the loop algorithm actively selecting selection 
%queries of size N_S and updating the posterior according to the simulated
%human answers. The results are saved in 'filename.....mat'
    % Fields in params:
    %   .init_file           % .mat file with all variables to load
    %   .N_iterations
    %   .human_model_type    % 'discrete_cdf' or 'gaussian_score'
    %   .progress_every
    %   .item_selection      % true (item selection queries) or false (labeling queries)
    %   .active              % true (active item selection) or false (random)

    % Load information from dataset
    load(params.init_file); 
    % Check that we correctly loaded required values
    req = {'C_init','theta','d','list_embeddings','y_train', 'scale', 'noise_factor'};
    for k = 1:numel(req)
        assert(exist(req{k},'var')==1, ['Missing variable in ', params.init_file, ': ', req{k}]);
    end
    switch params.human_model_type
        case 'discrete_cdf'   % AVA
            req_hm = {'score_votes','threshold_score'};
        case 'gaussian_score' % SocialSent
            req_hm = {'list_score','list_var'};
        otherwise
            error('Unknown human_model_type: %s', params.human_model_type);
    end
    for k = 1:numel(req_hm)
        assert(exist(req_hm{k},'var')==1, ...
            ['Missing variable in ', params.init_file, ': ', req_hm{k}]);
    end

    N_iterations = params.N_iterations;
    if ~exist('xtrain', 'var')
        xtrain = double(list_embeddings);
    end
    if ~exist('threshold_score', 'var')
        threshold_score = 0;
    end
    N_items = size(list_embeddings, 2);
    mu_init    = zeros(d+1, 1);
    mu_save    = zeros(d+1, N_iterations);
    C_diag_norm = zeros(N_iterations+1, 1);
    MSE_MT      = zeros(N_iterations+1, 1);
    accu        = zeros(N_iterations+1, 1);
    max_score   = zeros(N_iterations, 1);

    if params.item_selection == false
        N_S = 1;
        params.active = false;
    end

    mu = mvnrnd(mu_init, C_init)'; 
    sigma = C_init;
    C_diag_norm(1) = C_init(1,1);
    MSE_MT(1) = vecnorm(theta - mu)^2;
    accu(1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / (2*length(y_train));
    try
        for ii = 1:N_iterations
            mu_save(:, ii) = mu;
            % Select Query at random. 1: q_pos, 2: q_neg
            query_selected = randi([1,2]);
    
            % Item selection
            if params.active
                [S, ind_S, list_embeddings_remaining] = ...
                    select_first_word_Hsubstraction(list_embeddings, mu, sigma, noise_factor);
                for loop = 2:N_S
                    [S, ind_S, list_embeddings_remaining, max_sc] = ...
                        select_next_word_Hsubstraction(S, ind_S, list_embeddings_remaining, ...
                                                 mu, sigma, noise_factor, scale, query_selected);       
                end
                max_score(ii) = max_sc;
            else
                ind_S = randperm(N_items, N_S);
                S = list_embeddings(:, ind_S);
                [EH, HE] = get_expected_score(S,mu, sigma, ...
                                              noise_factor, scale, query_selected);
                max_score(ii) = HE - EH;
            end
    
            % Human model
            switch params.human_model_type
                case 'discrete_cdf'   % AVA case, uses score_votes, threshold_score
                    cdfs = cumsum(score_votes(:, ind_S));
                    rand_nums = rand(1, length(ind_S));
                    sample_score = sum(rand_nums > cdfs, 1) + 1;
    
                case 'gaussian_score' % SocialSent case, uses list_score, list_var
                    sample_score = list_score(ind_S) + ...
                                   randn(N_S, 1).*sqrt(list_var(ind_S));
    
                otherwise
                    error('Unknown human_model_type');
            end                    
            
            % Select item according to query
            if query_selected==1
                [score_x, ind_x] = max(sample_score);
            else
                [score_x, ind_x] = min(sample_score);
            end                     
    
            % Label item
            if score_x > threshold_score
                y =1;
            else
                y = -1;
            end

            % Update given label
            [mu, sigma] = update_given_y_logistic(S(:, ind_x).*noise_factor, ...
                                                  y, sigma, mu);   
    
            % Update given item selected       
            if params.item_selection
                if query_selected==1
                    [mu, sigma] = update_given_x(mu, sigma, scale.*S, ind_x);
                else
                    [mu, sigma] = update_given_x(mu, sigma, -1*scale.*S, ind_x);
                end
            end        
    
            C_diag_norm(ii+1) = prod(nthroot(svd(sigma), d+1));
            MSE_MT(ii+1) = vecnorm(theta - mu)^2;
            accu(ii+1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / ...
                            (2*length(y_train));                                    
    
            if mod(ii, params.progress_every)==0
                disp(['Iteration ', num2str(ii), '/', num2str(N_iterations), ...
                      ' in initialization ', num2str(init)])
            end
        end
    catch ME
        fprintf('Error in worker %d: %s\n', spmdIndex, ME.message);
        fprintf('Iteration %d', ii)
    end

    end_mu = mu;
    end_sigma = sigma;

    outname = filename + "_selection_S" + num2str(N_S) + "_init" + num2str(init) + ".mat";              
    save(outname, 'end_mu', 'end_sigma', 'mu_save', 'C_diag_norm', ...
         'MSE_MT', 'accu', 'max_score');      

end
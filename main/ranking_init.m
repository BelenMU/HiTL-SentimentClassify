function ranking_init(filename,init, N_S)
%ALGORITHM HUMAN IN THE LOOP WITH RANKING INIT Runs human in the loop
%algorithm actively selecting ranking queries of size N_S and updating the
%posterior according to the simulated human answers. The results are saved
%in 'filename_init#.mat'
    %% Initialize
    load("init_SocialSent_freq.mat");
    N_iterations = 400;
    noise_factor = 12; % Of the label given the example

    end_mu = zeros(d+1, 1);
    end_sigma = zeros(d+1, d+1);
    mu_save = zeros(d+1, N_iterations);
    C_diag_norm = zeros(N_iterations+1, 1);
    MSE_MT = zeros(N_iterations+1, 1);
    accu = zeros(N_iterations+1, 1);
    max_score = zeros(N_iterations, 1);

    % Initializevector to hold indices
    indices_vector = zeros(N_S, N_iterations);


    mu = mvnrnd(mu_init, C_init)';
    sigma = C_init;
    C_diag_norm(1) = C_init(1, 1);
    MSE_MT(1) = vecnorm(theta - mu)^2;
    accu(1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / (2*length(y_train));

    %% Run algorithm
    for ii = 1:N_iterations
        mu_save(:, ii) = mu;
        % Select set of examples to show the teacher
        [S, ind_S, list_embeddings_remaining] = select_first_word_Hsubstraction(list_embeddings, mu, sigma, noise_factor);
        for loop = 2:N_S
            [S, ind_S, list_embeddings_remaining, max_score] = select_next_word_ranking(S, ...
                                                        ind_S, list_embeddings_remaining, mu, sigma, ...
                                                        noise_factor, scale);
        end

        %Save indices from this iteration
        indices_vector(:, ii) = ind_S(:);
       

	 max_score(ii) = max_score;
        % Model Human
        % Sample Score a human would give to each of the words
        sample_score = list_score(ind_S) + randn(N_S, 1) .* sqrt(list_var(ind_S));
        % Rank from highest to lowest implicit score
        [~, ind_ranking] = sort(sample_score, 'descend');
        % Label words
        y = ones(N_S, 1);
        y(sample_score < 0) = -1;
        % Update given all labels
        for ind_x = 1:N_S
            [mu, sigma] = update_given_y_logistic(S(:, ind_x).*noise_factor, y(ind_x), sigma, mu);
        end
        % Update given word ranking   
        [mu, sigma] = update_given_S(mu, sigma, scale.*S, ind_ranking);

        C_diag_norm(ii+1) = prod(nthroot(svd(sigma), d+1));
        MSE_MT(ii+1) = vecnorm(theta - mu)^2;
        accu(ii+1) = 1 - sum(abs(sign(mu' * xtrain) - y_train)) / (2*length(y_train));

        if mod(ii, 200)==0
            disp(['Iteration ', num2str(ii), '/', num2str(N_iterations), ' in initialization ', num2str(init)])
        end
    end
    % Save Filename


    filename_init = filename + "_S" + num2str(N_S) + "_init" + num2str(init) + ".mat";
    disp("naming file");


    save(filename_init, 'end_mu', 'end_sigma', 'mu_save', 'C_diag_norm', 'MSE_MT', 'accu', 'max_score', 'indices_vector')
    disp("saving file");
end

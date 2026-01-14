function word_sentiment_selection(filename, N_initializations, N_S)
    params.init_file       = "init_word_sentiment.mat";
    params.N_iterations    = 2e3;
    params.human_model_type = 'gaussian_score';
    params.progress_every  = ceil(params.N_iterations / 10);
    params.word_selection = true;
    params.item_selection = true;
    params.active = true;

    if isempty(gcp('nocreate'))
        parpool;
    end
    parfor init = 1:N_initializations
        selection_init(filename, init, N_S, params);
    end
end

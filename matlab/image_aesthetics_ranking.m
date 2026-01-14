function image_aesthetics_ranking(filename, N_initializations, N_S)
    params.init_file       = "init_image_aesthetics.mat";
    params.N_iterations    = 2e3;
    params.human_model_type = 'discrete_cdf';
    params.progress_every  =  ceil(params.N_iterations / 10);
    params.active = true;

    if isempty(gcp('nocreate'))
        parpool;
    end    
    parfor init = 1:N_initializations
        ranking_init(filename, init, N_S, params);
    end
end

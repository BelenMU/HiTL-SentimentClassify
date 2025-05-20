filename = 'rank';
num_initializations = 5;
N_S = 7;

if isempty(gcp('nocreate'))
    parpool;
end
parfor init = 1:num_initializations
    ranking_init(filename,init,N_S)
end

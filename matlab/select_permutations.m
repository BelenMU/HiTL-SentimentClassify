function selected_perms = select_permutations(N_prev, top_permutation, max_permutations)

    % Check if all permutations < max_permutations
    num_perms = factorial(N_prev + 1);
    if num_perms <= max_permutations
        selected_perms = perms(1:(N_prev + 1));
        return;
    end

    % Initialize the permutations storage with top_permutation
    permutations = repmat(top_permutation, N_prev + 1, 1);

    % Generate all permutations where two indices are swapped from top_permutation
    for ii = 1:N_prev
        permutations(ii+1, ii) = top_permutation(ii+1);
        permutations(ii+1, ii+1) = top_permutation(ii);
    end

    % Determine the number of permutations to sample randomly (if necessary)
    remaining_slots = max_permutations - size(permutations, 1);
    if remaining_slots > 0
        permutations_set = unique(permutations, 'rows');
        if num_perms <= 1e3
            % Generate all permutations
            ranking_options = perms(1:(N_prev + 1));
            remaining_perms = setdiff(ranking_options, permutations_set, 'rows');
            random_indices = randperm(size(remaining_perms, 1), remaining_slots);
            sampled_perms = remaining_perms(random_indices, :);
            selected_perms = [permutations_set; sampled_perms];
        else
            while size(permutations_set, 1) < max_permutations
                % Generate a random permutation
                random_perm = randperm(N_prev + 1);
                % Add the random permutation if it's not already in the set
                if ~ismember(random_perm, permutations_set, 'rows')
                    permutations_set = [permutations_set; random_perm];
                end
            end
            selected_perms =permutations_set;
        end
    else
        % If there are more than enough interleaved_perms, truncate the list
        selected_perms = permutations(1:max_permutations, :);
    end
%{
    % Ensure no repeated permutations and trim to the exact needed size again
    selected_perms = unique(selected_perms, 'rows');
    if size(selected_perms, 1) > max_permutations
        selected_perms = selected_perms(1:max_permutations, :);
    end
%}
end
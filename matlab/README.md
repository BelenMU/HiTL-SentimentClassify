# MATLAB

This repository contains the MATLAB code for the paper titled "Beyond Labels: Information-Efficient Human-in-the-Loop Learning via Ranking and Selection Queries," authored by Belen Martin-Urcelay, Yoonsang Lee, Matthieu R. Bloch, and Christopher J. Rozell, 2026.

The code implements active query selection, and Bayesian posterior updates for both word sentiment and image aesthetics classification tasks. 

## Requirements

Matlab R2023a with 

- Optimization Toolbox version 9.4
- Parallel Computing Toolbox version 7.7

## Main scripts 

**`word_sentiment_selection.m`** and **`image_aesthetics_selection.m`** contain the implementation of Algorithm 2 for the word sentiment and image aesthetic classification tasks respectively. The code contains the human in the loop logic for :

- **Selection questions** of the form: “Select and label the most positive/negative item in the list.”
- **Labeling questions** of the form: “Label this item as positive or negative.”

Function inputs:

- filename: prefix of the name of the output .mat file with the results.
- N_initializations: Number of random initializations for which the algorithm is run, this will be ran in parallel.
- N_S: The size of the item set, $|\mathcal{S}|$, from which the human chooses an item.

Configuration parameters (set inside the function):

- params.N_iterations: How many human interactions the experiment should simulate.
- params.progress_every: How often (in number of interactions) the status should be printed.
- params.item_selection: Binary parameter, set to *true* to ask selection questions, and to *false* to ask labeling questions.
- params.active: Binary parameter, set to *true* to select items in the set actively, and to *false* to select items uniformly at random.

**`word_sentiment_ranking.m`** and **`image_aesthetics_ranking.m`** contain the implementation of Algorithm 3 for the word sentiment and image aesthetic classification tasks respectively. The code contains the human in the loop logic for ranking questions of the form $q_{\text{rank}} = $ ``Rank the items from highest to lowest score and indicate which is the last positive example in the ranked list.'' 

Function inputs:

- filename: prefix of the name of the output .mat file with the results.
- N_initializations: Number of random initializations for which the algorithm is run, this will be ran in parallel.
- N_S: The size of the item set, $|\mathcal{S}|$, from which the human chooses an item.

Configuration parameters (set inside the function):

- params.N_iterations: How many human interactions the experiment should simulate.
- params.progress_every: How often (in number of interactions) the status should be printed.
- params.active: Binary parameter, set to *true* to select items in the set actively, and to *false* to select items uniformly at random.

## Dataset information

The code expects dataset‑specific initialization `.mat` files. Here we include the corresponding files for the word sentiment and image aesthetic classification tasks shown in the paper. 

 **`init_word_sentiment`** contains all variables needed to run the word‑sentiment experiments (labeling, selection and ranking):

- list_words: List of words to classify
- list_embeddings: Word embedding obtained from the word2vec encoder. 
- list_score, list_var: Mean and variance of the intrinsic valence score of each word as provided by the SocialSent dataset (https://nlp.stanford.edu/projects/socialsent/). 
- x_train, y_train: Embedding and labels of words whose probability of being positive is outside the range $[0.4,0.6]$. This way, we remove the neutral words when measuring accuracy. 
- theta: MMSE classifier between positive and negative words. 
- noise_factor, scale: scalars to account for noise level.
- d: dimension of the word embeddings from word2vec (300)
- C_init: Prior of the covariance matrix, computed as the covariance of uniformly sampled vectors.

 **`init_image_aesthetics`** contains all variables needed to run the image aesthetic experiments (labeling, selection and ranking):

- list_embeddings: Image embeddings obtained from the ViT-L/14 pretrained on CLIP for 21,979 landscape images in the AVA dataset.
- score_votes:  Distribution of scores from human raters from 1 to 10, according to the AVA dataset.
- theshold_score:  Threshold separating images high high vs. low aesthetic score. It is given by the median of the averages 5.48.
- y_train: Labels of images as positive (1) or negative (-1). 
- theta: MMSE classifier between positive and negative images. 
- noise_factor, scale: scalars to account for noise level.
- d: dimension of the word embeddings from the ViT-L/14  (768)
- C_init: Prior of the covariance matrix, computed as the covariance of uniformly sampled vectors.

## Auxiliary functions

- `selection_init.m`: Core per‑initialization loop for **labeling** and **selection** (Algorithm 2) queries. It loads the dataset initialization, configures the human model and item selection mode (active vs. random), runs the iterative human‑in‑the‑loop updates, and saves posterior trajectories and metrics over iterations for one initialization.
- `ranking_init.m`: Core per‑initialization loop for **ranking** (Algorithm 3) queries. It loads the dataset initialization, configures the human model for ranking queries and item selection mode (active vs. random), runs the iterative human‑in‑the‑loop updates, and saves posterior trajectories and metrics over iterations for one initialization.
- `select_first_word_Hsubstraction.m`: Actively selects first item in the set by maximizing the active learning heuristic.
- `select_next_word_Hsubstraction.m`: Greedily selects the next item to add to the set by maximizing the active learning heuristic for selection questions.
- `select_next_word_ranking.m`: Greedily selects the next item to add to the set by maximizing the active learning heuristic for ranking questions.
- `select_permutations.m`: It enumerates all permutations when feasible, and falls back to a sampled subset when the combinatorics are too large, to keep the algorithm computationally tractable.
- `get_expected_score.m`: Provides the value of the active learning heuristic when the item set is selected at random for selection queries.
- `update_given_y_logistic.m`: Approximates the posterior given the label iterating over lines 5, 6, 7 of of Algorithm 4 in the paper.
- `update_given_x.m`: Approximates the posterior given the item selected. It leverages
  - `compute_elbo_bound.m`: Computes the upper bound of the ELBO in eq. (8)
  - `fh.m`: Computes output and gradient of objective function in eq. (8) for covariance update.
- `update_given_S.m`: Approximates the posterior given the item ranking. It applies `update_given_x.m` recursively.

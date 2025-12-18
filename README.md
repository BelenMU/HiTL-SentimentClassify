# Beyond Labels: Information-Efficient Human-in-the-Loop Learning via Ranking and Selection Queries

This repository contains the MATLAB and Python code for the paper titled "Beyond Labels: Information-Efficient Human-in-the-Loop Learning via Ranking and Selection Queries," authored by Belen Martin-Urcelay, Yoonsang Lee, Matthieu R. Bloch, and Christopher J. Rozell, 2024.

## Abstract

Integrating human expertise into machine learning systems often reduces the role of experts to labeling oracles, a paradigm that limits the amount of information exchanged and fails to capture the nuance of human judgment. We propose a human-in-the-loop framework to learn binary classifiers with richer query types, specifically item ranking and exemplar selection. We introduce probabilistic human response models for these rich queries based on the relationship observed between the perceived implicit score of an item and the distance from its embedding to the unknown classifier. With these models, we design active learning algorithms that leverage the rich queries to increase the information gained per interaction. We provide theoretical bounds on sample complexity and develop a tractable and computationally efficient variational approximation for Bayesian updates. We further extend active learning strategies to select queries that maximize information rate, explicitly balancing informational value against annotation cost. Through experiments with simulated annotators derived from crowdsourced word-sentiment and image-aesthetic datasets, we demonstrate significant reductions on sample complexity. Our cost-aware algorithm in the word sentiment classification task reduces annotation time by more than 57\% compared to traditional label-only active learning. 

## Requirements

Matlab R2023a with 

- Optimization Toolbox version 9.4
- Parallel Computing Toolbox version 7.7

## MATLAB
The folder 'matlab' contains the implementation of the Algorithm in MATLAB.

'main.m' contains the implementation of Algorithm 2. To configure the various setting, the following parameters in 'main.m' should be set:

- num_S: Defines the size of the word set, $|\mathcal{S}|$, from which the human chooses a word.
- word_selection: Set to false for traditional role of human of only providing labels. Set to true to also request a word selection.
- active: Set to true to actively select the word set in each query. Set to false to select the word set randomly.

The remaining functions are auxiliary 

- select_first_word_Hsubstraction.m: Actively selects first word in the word set by maximizing the active learning heuristic.
- select_next_word_Hsubstraction.m: Greedily selects the next word to add to the word set by maximizing the active learning heuristic.
- get_expected_score.m: Provides the value of the active learning heuristic when the word set is selected at random.
- update_given_y_logistic.m: Approximates the posterior given the label iterating over eq. (5), (6) and (7) in the paper.
- update_given_x.m: Approximates the posterior given the word selected. It leverages
  - compute_elbo_bound.m: Computes the upper bound of the ELBO in eq. (8)
  - fh.m: Computes output and gradient of objective function in eq. (9) for covariance update.

init_SocialSent_freq contains data from SocialSent (https://nlp.stanford.edu/projects/socialsent/) from which we simulate human answers.

## Python
The folder 'python' contains an equivalent implementation of the code in Python. It reproduces the main algorithm and experiments from the MATLAB version using Python libraries.

## Replicating Figures

#### Figure 1

Figure 2 in the paper is the basis for our likelihood models. To replicate the figure run *'analysis_distance_vs_valence.m'*. Note that the human scores are taken from the SocialSent dataset (https://nlp.stanford.edu/projects/socialsent/)

- list_adj2000: Dataset of the most used adjectives in the decade of the 2000s.
- list_freq2000: Dataset of the most used words in the decade of the 2000s.

#### Figure 3

Figure 3 in the paper shows the results of Algorithm 2. To replicate the figure we first need to run Algorithm 2 with the main code. Next, 

- 'plot_with_errorbars.m' creates a figure in which the mean over initializations is shown as a solid line and the standard error as the shaded area that surround it. Use it to recreate Figure 3a, 3c and 3d.
- 'plot_accuracy_vs_margin' creates a figure analogous to Figure 3b. In which the accuracy between the estimated classifier and the ground truth are compared, for difference dictionary sizes by discarding neutral words.

## Contact Information 

Email: burcelay3@gatech.edu



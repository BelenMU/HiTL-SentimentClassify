# Beyond Labels: Information-Efficient Human-in-the-Loop Learning via Ranking and Selection Queries

This repository contains the MATLAB and Python code for the paper titled "Beyond Labels: Information-Efficient Human-in-the-Loop Learning via Ranking and Selection Queries," authored by Belen Martin-Urcelay, Yoonsang Lee, Matthieu R. Bloch, and Christopher J. Rozell, 2026.

## Abstract

Integrating human expertise into machine learning systems often reduces the role of experts to labeling oracles, a paradigm that limits the amount of information exchanged and fails to capture the nuance of human judgment. We propose a human-in-the-loop framework to learn binary classifiers with richer query types, specifically item ranking and exemplar selection. We introduce probabilistic human response models for these rich queries based on the relationship observed between the perceived implicit score of an item and the distance from its embedding to the unknown classifier. With these models, we design active learning algorithms that leverage the rich queries to increase the information gained per interaction. We provide theoretical bounds on sample complexity and develop a tractable and computationally efficient variational approximation for Bayesian updates. We further extend active learning strategies to select queries that maximize information rate, explicitly balancing informational value against annotation cost. Through experiments with simulated annotators derived from crowdsourced word-sentiment and image-aesthetic datasets, we demonstrate significant reductions on sample complexity. Our cost-aware algorithm in the word sentiment classification task reduces annotation time by more than 57\% compared to traditional label-only active learning. 

## Requirements

Matlab R2023a with 

- Optimization Toolbox version 9.4
- Parallel Computing Toolbox version 7.7

## MATLAB
The folder 'matlab' contains the implementation of the Algorithms 2 and 3 in MATLAB for the word sentiment and image aesthetic classification tasks. It also reproduces the experiments that support the human model in Assumption 1. This is the main code implementation.

## Python
The folder 'python' contains an equivalent implementation of the code in Python. It reproduces the main algorithm and experiments for the word sentiment experiments from the MATLAB version using Python libraries.

## Contact Information 

Email: burcelay3@gatech.edu





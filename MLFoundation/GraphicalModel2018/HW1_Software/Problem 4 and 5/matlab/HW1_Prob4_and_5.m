% ==================================================================
% Homework 1, Problem 4 and 5
% Center for Informatics and Computational Sciences
% University of Notre Dame
% ==================================================================
clc;
clear;

addpath(genpath('graph'))   % add the current path
addpath(genpath('Inference_engine'))
addpath(genpath('tabular_CPD'))


N = 4;                          % Number of nodes
dag = zeros(N,N);               % initializing the adjacency matrix
C = 1; S = 2; R = 3; W = 4;     % Nodes
dag(C,[R S]) = 1;               % Stting adjacency matrix for C->R and C->S
dag(R,W) = 1;                   % Setting adjacency matrix for R -> W
dag(S,W)=1;                     % Setting adjacency matrix for S -> W
ns = 2*ones(1,N);               % Number of states of each nodes 
bnet = make_bayesnet(dag, ns);  % formulate the bayesnet structure

% Inputing the conditional probability distribution
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

% Initializing the variable_elimination_engine
eng = var_elim_inf_engine(bnet);
% Setting Evidence
evidence = cell(1,N);
evidence{W} = 2;
[eng, loglik] = enter_evidence(eng, evidence);
% Computing the marginals
marg = marginal_nodes(eng, S);

% Displaying the conditional probability
disp('===================================')
fprintf('P(s=0|w=1) = %1.4f\n',marg.T(1));
fprintf('P(s=1|w=1) = %1.4f\n',marg.T(2));
disp('===================================')
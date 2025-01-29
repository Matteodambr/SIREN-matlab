% Script used an example for function calls to generate SIREN network

addpath(genpath('src')) ;

% Example function call:
neuronNumber = 200 ;
actionNumber = 7 ;
observationNumber = 30 ;
omega_0 = 30 ; % Typical value from original paper that seems to work well for most applications
actorNet = SIREN_Lx3_NxXXX_stochastic(neuronNumber,actionNumber, observationNumber, omega_0) ;
analyzeNetwork(actorNet) ;

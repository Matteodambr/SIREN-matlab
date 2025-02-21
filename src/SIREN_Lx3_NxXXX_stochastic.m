function actorNet = SIREN_Lx3_NxXXX_stochastic(neuronNumber, actionNumber, observationNumber, omega_0)
% SIREN_Lx3_NxXXX_stochastic Creates a stochastic SIREN network
%
%   actorNet = SIREN_Lx3_NxXXX_stochastic(neuronNumber, actionNumber, observationNumber, omega_0)
%   creates a stochastic SIREN (Sinusoidal Representation Network) with the
%   specified number of neurons, actions, and observations. The network
%   uses sinusoidal activation functions and is initialized with the given
%   omega_0 scaling factor.
%
%   Inputs:
%       neuronNumber      - Number of neurons in each fully connected layer
%       actionNumber      - Number of actions (output size)
%       observationNumber - Number of observations (input size)
%       omega_0           - Scaling factor for the sinusoidal activation functions
%
%   Outputs:
%       actorNet          - Initialized SIREN network as a dlnetwork object
%
%   Example:
%       actorNet = SIREN_Lx3_NxXXX_stochastic(128, 4, 16, 30);
%
%   See also: dlnetwork, layerGraph, fullyConnectedLayer, functionLayer, softplusLayer
% Network structure:
%              +------------------+
%              |  Input Sequence  |
%              +------------------+
%                      |
%                      v
%             +-------------------+
%             | Feedforward Block |
%             |   (Dense Layer)   |
%             +-------------------+
%                       |
%            +----------+----------+
%            |                     |
%            v                     v
%    +----------------+    +---------------+
%    |   Mean Layers  |    |   Std Layers  |
%    +----------------+    +---------------+
%            |                      |
%            v                      v
%      +----------+            +----------+
%      |   Mean   |            |   Std    |
%      +----------+            +----------+
lgraph = layerGraph() ;

tempLayers = [
    featureInputLayer(observationNumber,"Name","observations")
    fullyConnectedLayer(neuronNumber,"Name","fc1")
    scalingLayer("Name", "omega_0", Scale=omega_0, Bias=0) ;
    functionLayer(@(x)sin(x),"Name","sineLayer","Acceleratable",true)] ;
lgraph = addLayers(lgraph,tempLayers) ;

tempLayers = [
    fullyConnectedLayer(neuronNumber,"Name","fc2_mean")
    functionLayer(@(x)sin(x),"Name","sineLayer_1","Acceleratable",true)
    fullyConnectedLayer(neuronNumber,"Name","fc3_mean")
    functionLayer(@(x)sin(x),"Name","sineLayer_3","Acceleratable",true)
    fullyConnectedLayer(actionNumber,"Name","fc4_mean")
    tanhLayer("Name","output_mean_tanh")] ;
lgraph = addLayers(lgraph,tempLayers) ;

tempLayers = [
    fullyConnectedLayer(neuronNumber,"Name","fc2_std")
    functionLayer(@(x)sin(x),"Name","sineLayer_2","Acceleratable",true)
    fullyConnectedLayer(neuronNumber,"Name","fc3_std")
    functionLayer(@(x)sin(x),"Name","sineLayer_4","Acceleratable",true)
    fullyConnectedLayer(actionNumber,"Name","fc4_std")
    softplusLayer("Name","output_std_softplus")] ;
lgraph = addLayers(lgraph,tempLayers) ;

% clean up helper variable
clear tempLayers ;

lgraph = connectLayers(lgraph,"sineLayer","fc2_mean") ;
lgraph = connectLayers(lgraph,"sineLayer","fc2_std") ;

% Convert layer graph to dlnetwork
actorNet = dlnetwork(lgraph) ;

% Initialize the SIREN network with the correct weights
actorNet = initializeSIREN(actorNet, 'fc1', omega_0) ;

end
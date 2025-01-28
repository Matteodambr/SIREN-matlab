% filepath: /d:/github_local/MDA_PhD-AI_SRGuidance/Functions/DRL-neural-networks_matlab/src/actor/SIREN-matlab/src/temp_test_network.m
% Test script for the imported PyTorch network

addpath('Aten');
addpath('utils');

% Import the network from the PyTorch model file
network = importNetworkFromPyTorch('traced_siren_network.pt');

% Define sample input data
inputSize = [1, 2]; % Adjust this size based on your network's input size
inputData = rand(inputSize); % Example input data

% Create a dlarray with the appropriate format
dlX1 = dlarray(inputData, 'SSCB'); % Replace 'SSCB' with the correct format

% Initialize the network
network = initialize(network, dlX1);

% Perform a forward pass
outputData = predict(network, dlX1);

% Display the input and output data
disp('Input Data:');
disp(inputData);

disp('Output Data:');
disp(outputData);

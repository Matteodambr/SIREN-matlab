% Test script for the imported PyTorch network

actNumber = 7 ;
obsNumber = 30 ;

addpath('Aten_custom') ;
addpath('utils') ;

% Import the network from the PyTorch model file
network = importNetworkFromPyTorch('traced_siren_network.pt') ;  % Network cannot be automatically initialized, since placeholder functions are generated. Network needs to be initialized after the functions have been replaced.

% Replace auto-generated placeholder functions with the custom ones
% TODO: This function must be run from the SIREN-matlab folder
replaceSIRENPlaceholderFunctions ;

% Define sample input data
% TODO: Add realistic input data generated from importPARAM
inputData = rand(obsNumber, 1) ; % Example input data

% Create a dlarray with the appropriate format
dlX1 = dlarray(inputData, 'CB') ; % 'CB' format for a single vector input

% Initialize the network
network = initialize(network, dlX1) ;

% Define the target data (dlY) for the loss function
dlY = rand(7, 1)' ; % Replace with your target data

% Test network with a forward pass
try
    outputData = predict(network, dlX1);
catch ME
    disp('Error during forward pass:');
    disp(ME.message);
    rethrow(ME);
end

% Verify the output size
assert(isequal(size(outputData), [1, actNumber]), 'Output size does not match actNumber');

% Display the input and output data
disp('Input Data:');
disp(inputData);

disp('Output Data:');
disp(outputData);

% Compute gradients with respect to the loss
try
    [loss, gradients] = dlfeval(@computeLoss, network, dlX1, dlY);
    disp('Gradients computed successfully.');
catch ME
    disp('Error during backward pass:');
    disp(ME.message);
    rethrow(ME);
end

% Display the gradients
disp('Gradients:');
disp(gradients);

% Define a dummy loss function for testing
function [loss, gradients] = computeLoss(network, dlX1, dlY)
    outputData = predict(network, dlX1);
    loss = mean((outputData - dlY).^2, 'all');
    gradients = dlgradient(loss, network.Learnables);
end

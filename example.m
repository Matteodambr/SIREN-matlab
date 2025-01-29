% Script used an example for function calls to generate SIREN network


% Inputs
input_features = 30 ; % Number of inputs to the network (observations)
output_features = 7 ; % Number of outputs of the network (actions, in case of stochastic actor, standard deviation outputs are automatically computed)
hidden_features = 200 ; % Number of neurons in each layer
hidden_layers = 3 ; % Number of layers in the network
outermost_linear = true ;

% Generate SIREN network
network = genSIRENnetwork(input_features, output_features, hidden_features, hidden_layers, outermost_linear)
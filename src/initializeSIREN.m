function sirenNet = initializeSIREN(sirenNet, firstLayerName, omega_0)
%INITIALIZESIREN Initialize the SIREN network with specific weights
%   sirenNet = INITIALIZESIREN(sirenNet, firstLayerName, omega_0) initializes
%   the weights of the SIREN network. The first layer is initialized differently
%   from the other layers.
%
%   Inputs:
%       sirenNet - The SIREN network to be initialized
%       firstLayerName - The name of the first layer in the network
%       omega_0 - The scaling factor for the first layer's weights
%
%   Outputs:
%       sirenNet - The initialized SIREN network
%
%   This function follows the initialization procedure described in Section 3.2
%   of the paper "Implicit Neural Representations with Periodic Activation
%   Functions" (https://arxiv.org/abs/2006.09661).

% Define how many learnable arrays are present in the network
num_learnables = size(sirenNet.Learnables, 1) ;

% Loop through each learnable array, to initialize it
for k = 1:num_learnables

    % Initialize only the weights of each layer
    if strcmp(sirenNet.Learnables(k,2).Parameter, 'Weights')

        % Check what layer is being initialized (first layer is initialized differently)
        isFirstLayer = strcmp(sirenNet.Learnables(k,1).Layer, firstLayerName) ;

        % Number of weights that need to be generated for the specific layer
        [rows, cols] = size(sirenNet.Learnables(k,3).Value{1}) ;

        % Retrieve number of inputs to the current layer
        input_feature_number = cols ; % W*x --> second dimension of W corresponds to the length of x, i.e. the number of inputs. 

        switch isFirstLayer

            case 1 % Initialization of the first layer

                % First layer initialization from uniform distribution
                maxval = 1/input_feature_number ;
                minval = -maxval ;
                newWeights = minval + (maxval-minval) * rand(rows, cols) ; % Generate MxN array of initialized weights for layer

                % Replace weights in network with new ones, of type single
                sirenNet.Learnables(k,3).Value{1} = dlarray(single(newWeights)) ;

            case 0 % Initialization of all subsequent SIREN layers

                % Non-first layer initialization from uniform distribution
                maxval = sqrt(6/input_feature_number) / omega_0 ;
                minval = -maxval ;
                newWeights = minval + (maxval-minval) * rand(rows, cols) ; % Generate MxN array of initialized weights for layer

                % Replace weights in network with new ones, of type single
                sirenNet.Learnables(k,3).Value{1} = dlarray(single(newWeights)) ;

        end

    end

end

end
function varargout = pyAtenSin(varargin)
% Function for the PyTorch operator named aten::sin.

import traced_siren_network.ops.*

inputs = cell(1,nargin);
[inputs{:}] = permuteToPyTorchDimensionOrder(varargin{:});

outputs = cell(1,nargout);

% Implement the sin operation
for i = 1:nargin
    X = inputs{i};
    % Ensure X.value is a dlarray
    if ~isa(X.value, 'dlarray')
        X.value = dlarray(X.value);
    end
    Y = sin(X.value);
    outputs{i} = struct('value', Y, 'rank', X.rank);
end

varargout = cell(1,nargout);
[varargout{:}] = permutePyTorchToReversePyTorch(outputs{:});
end
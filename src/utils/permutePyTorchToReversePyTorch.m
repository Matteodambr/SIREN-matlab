function varargout = permutePyTorchToReversePyTorch(varargin)
    % Permute dimensions back from PyTorch's order to MATLAB's order
    
    varargout = cell(size(varargin));
    for i = 1:nargin
        varargout{i} = permute(varargin{i}, [3, 4, 2, 1]); % Example permutation
    end
    end
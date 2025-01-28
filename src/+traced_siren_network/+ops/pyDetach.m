function dlY = pyDetach(dlX)
    %PYDETACH Returns a new dlarray, detached from the current graph.
    %   at::Tensor at::detach(const at::Tensor &self)
    
    %   Copyright 2023 The MathWorks, Inc.

    import traced_siren_network.ops.*

    Xval = dlX.value;

    Xdim = dims(Xval);
    Yrank = dlX.rank;

    % create a new dlarray
    Yval = extractdata(Xval);

    Yval = dlarray(Yval, Xdim);
    dlY = struct('value', Yval, 'rank', Yrank);
    
end
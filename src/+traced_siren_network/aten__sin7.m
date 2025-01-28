classdef aten__sin7 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch
    %aten__sin7 Auto-generated custom layer
    % Auto-generated by MATLAB on 28-Jan-2025 23:37:46
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        
        
        
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        
    end
    
    methods
        function obj = aten__sin7(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [sin_10, sin_10_rank] = predict(obj,sin_8, sin_8_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [sin_8, sin_8_format] = permuteToReversePyTorch(sin_8, '', numel(sin_8_rank));
            [sin_8] = struct('value', sin_8, 'rank', int64(numel(sin_8_rank)));
            
            % Placeholder function for aten::sin.
            [sin_10] = pyAtenSin(sin_8);
            [sin_10_rank] = ones([1,sin_10.rank], 'single');
            sin_10_rank = dlarray(sin_10_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(sin_10.value) == 'U'))
                sin_10 = permute(sin_10.value, fliplr(1:max(2,sin_10.rank)));
            end
            
        end
        
        
        
        function [sin_10, sin_10_rank] = forward(obj,sin_8, sin_8_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [sin_8, sin_8_format] = permuteToReversePyTorch(sin_8, '', numel(sin_8_rank));
            [sin_8] = struct('value', sin_8, 'rank', int64(numel(sin_8_rank)));
            
            % Placeholder function for aten::sin.
            [sin_10] = pyAtenSin(sin_8);
            [sin_10_rank] = ones([1,sin_10.rank], 'single');
            sin_10_rank = dlarray(sin_10_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(sin_10.value) == 'U'))
                sin_10 = permute(sin_10.value, fliplr(1:max(2,sin_10.rank)));
            end
            
        end
        
        
    end
end


classdef aten__linear8 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch
    %aten__linear8 Auto-generated custom layer
    % Auto-generated by MATLAB on 29-Jan-2025 11:26:00
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        
        
        
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        Param_weight
        Param_bias
    end
    
    methods
        function obj = aten__linear8(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [linear_9, linear_9_rank] = predict(obj,linear_argument1_1, linear_argument1_1_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [linear_argument1_1, linear_argument1_1_format] = permuteToReversePyTorch(linear_argument1_1, '', numel(linear_argument1_1_rank));
            [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', int64(numel(linear_argument1_1_rank)));
            
            linear_weight_1 = obj.Param_weight;
            
            [linear_weight_1] = struct('value', linear_weight_1, 'rank', 2);
            
            linear_bias_1 = obj.Param_bias;
            
            [linear_bias_1] = struct('value', linear_bias_1, 'rank', 1);
            
            [linear_9] = pyLinear(linear_argument1_1, linear_weight_1, linear_bias_1);
            [linear_9_rank] = ones([1,linear_9.rank], 'single');
            linear_9_rank = dlarray(linear_9_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(linear_9.value) == 'U'))
                linear_9 = permute(linear_9.value, fliplr(1:max(2,linear_9.rank)));
            end
            
        end
        
        
        
        function [linear_9, linear_9_rank] = forward(obj,linear_argument1_1, linear_argument1_1_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [linear_argument1_1, linear_argument1_1_format] = permuteToReversePyTorch(linear_argument1_1, '', numel(linear_argument1_1_rank));
            [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', int64(numel(linear_argument1_1_rank)));
            
            linear_weight_1 = obj.Param_weight;
            
            [linear_weight_1] = struct('value', linear_weight_1, 'rank', 2);
            
            linear_bias_1 = obj.Param_bias;
            
            [linear_bias_1] = struct('value', linear_bias_1, 'rank', 1);
            
            [linear_9] = pyLinear(linear_argument1_1, linear_weight_1, linear_bias_1);
            [linear_9_rank] = ones([1,linear_9.rank], 'single');
            linear_9_rank = dlarray(linear_9_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(linear_9.value) == 'U'))
                linear_9 = permute(linear_9.value, fliplr(1:max(2,linear_9.rank)));
            end
            
        end
        
        
    end
end


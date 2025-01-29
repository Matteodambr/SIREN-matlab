classdef aten__mul12 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch
    %aten__mul12 Auto-generated custom layer
    % Auto-generated by MATLAB on 29-Jan-2025 10:50:52
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        mul_7
        
        
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        
    end
    
    methods
        function obj = aten__mul12(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [mul_8, mul_8_rank] = predict(obj,mul_6, mul_6_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [mul_6, mul_6_format] = permuteToReversePyTorch(mul_6, '', numel(mul_6_rank));
            [mul_6] = struct('value', mul_6, 'rank', int64(numel(mul_6_rank)));
            
            [mul_7] = makeStructForConstant(single(obj.mul_7), int64([0]), "Tensor");
            [mul_8] = pyElementwiseBinary(mul_6, mul_7, 'times');
            [mul_8_rank] = ones([1,mul_8.rank], 'single');
            mul_8_rank = dlarray(mul_8_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(mul_8.value) == 'U'))
                mul_8 = permute(mul_8.value, fliplr(1:max(2,mul_8.rank)));
            end
            
        end
        
        
        
        function [mul_8, mul_8_rank] = forward(obj,mul_6, mul_6_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [mul_6, mul_6_format] = permuteToReversePyTorch(mul_6, '', numel(mul_6_rank));
            [mul_6] = struct('value', mul_6, 'rank', int64(numel(mul_6_rank)));
            
            [mul_7] = makeStructForConstant(single(obj.mul_7), int64([0]), "Tensor");
            [mul_8] = pyElementwiseBinary(mul_6, mul_7, 'times');
            [mul_8_rank] = ones([1,mul_8.rank], 'single');
            mul_8_rank = dlarray(mul_8_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(mul_8.value) == 'U'))
                mul_8 = permute(mul_8.value, fliplr(1:max(2,mul_8.rank)));
            end
            
        end
        
        
    end
end


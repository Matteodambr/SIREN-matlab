classdef aten__detach1 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch
    %aten__detach1 Auto-generated custom layer
    % Auto-generated by MATLAB on 29-Jan-2025 10:50:52
    
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
        function obj = aten__detach1(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [detach_input_1, detach_input_1_rank] = predict(obj,detach_6, detach_6_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [detach_6, detach_6_format] = permuteToReversePyTorch(detach_6, '', numel(detach_6_rank));
            [detach_6] = struct('value', detach_6, 'rank', int64(numel(detach_6_rank)));
            
            [detach_input_1] = pyDetach(detach_6);
            [detach_input_1_rank] = ones([1,detach_input_1.rank], 'single');
            detach_input_1_rank = dlarray(detach_input_1_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(detach_input_1.value) == 'U'))
                detach_input_1 = permute(detach_input_1.value, fliplr(1:max(2,detach_input_1.rank)));
            end
            
        end
        
        
        
        function [detach_input_1, detach_input_1_rank] = forward(obj,detach_6, detach_6_rank)
            import traced_siren_network.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [detach_6, detach_6_format] = permuteToReversePyTorch(detach_6, '', numel(detach_6_rank));
            [detach_6] = struct('value', detach_6, 'rank', int64(numel(detach_6_rank)));
            
            [detach_input_1] = pyDetach(detach_6);
            [detach_input_1_rank] = ones([1,detach_input_1.rank], 'single');
            detach_input_1_rank = dlarray(detach_input_1_rank,'UU');
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(detach_input_1.value) == 'U'))
                detach_input_1 = permute(detach_input_1.value, fliplr(1:max(2,detach_input_1.rank)));
            end
            
        end
        
        
    end
end


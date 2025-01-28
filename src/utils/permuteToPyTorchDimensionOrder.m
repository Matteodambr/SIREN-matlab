function permutedTensor = permuteToPyTorchDimensionOrder(tensor)
    % Assuming the input tensor is in the format [height, width, channels, batch]
    % and we want to convert it to [batch, channels, height, width]
    permutedTensor = permute(tensor, [4, 3, 1, 2]);
end
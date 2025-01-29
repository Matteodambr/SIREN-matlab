function actorNet = SIREN_Lx3_NxXXX_stochastic(neuronNumber, actionNumber, observationNumber, omega_0)

% TODO: add header
% BUG: Check whether the neural network initialization is done again
% outside of this function, otherwise it will be overwritten.

lgraph = layerGraph() ;

tempLayers = [
    featureInputLayer(observationNumber,"Name","observations")
    fullyConnectedLayer(neuronNumber,"Name","fc1")
    scalingLayer("Name", "omega_0", Scale=omega_0, Bias=0) ;
    functionLayer(@(x)sin(x),"Name","sineLayer","Acceleratable",true)] ;
lgraph = addLayers(lgraph,tempLayers) ;

tempLayers = [
    fullyConnectedLayer(neuronNumber,"Name","fc2_mean")
    functionLayer(@(x)sin(x),"Name","sineLayer_1","Acceleratable",true)
    fullyConnectedLayer(neuronNumber,"Name","fc3_mean")
    functionLayer(@(x)sin(x),"Name","sineLayer_3","Acceleratable",true)
    fullyConnectedLayer(actionNumber,"Name","fc4_mean")
    functionLayer(@(x)sin(x),"Name","output_mean_sin","Acceleratable",true)] ;
lgraph = addLayers(lgraph,tempLayers) ;

tempLayers = [
    fullyConnectedLayer(neuronNumber,"Name","fc2_std")
    functionLayer(@(x)sin(x),"Name","sineLayer_2","Acceleratable",true)
    fullyConnectedLayer(neuronNumber,"Name","fc3_std")
    functionLayer(@(x)sin(x),"Name","sineLayer_4","Acceleratable",true)
    fullyConnectedLayer(actionNumber,"Name","fc4_std")
    softplusLayer("Name","output_std_softplus")] ;
lgraph = addLayers(lgraph,tempLayers) ;

% clean up helper variable
clear tempLayers ;

lgraph = connectLayers(lgraph,"sineLayer","fc2_mean") ;
lgraph = connectLayers(lgraph,"sineLayer","fc2_std") ;

% Convert layer graph to dlnetwork
actorNet = dlnetwork(lgraph) ;

% Initialize the SIREN network with the correct weights
actorNet = initializeSIREN(actorNet, 'fc1', omega_0) ;

end
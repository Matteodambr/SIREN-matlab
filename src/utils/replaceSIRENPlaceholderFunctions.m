function replaceSIRENPlaceholderFunctions()

% Define the source and destination directories
sourceDir = 'src/Aten_custom' ;
destDir = '+traced_siren_network/+ops' ;

% List of functions to replace
functionsToReplace = {'pyAtenSin.m'} ;

% Loop through each function and replace it
for k = 1:length(functionsToReplace)

    sourceFile = fullfile(sourceDir, functionsToReplace{k}) ;
    destFile = fullfile(destDir, functionsToReplace{k}) ;

    % Copy the file from source to destination
    copyfile(sourceFile, destFile) ;

    % Display replace files
    disp(['Replaced Aten network function ', fullfile(destDir, functionsToReplace{k}), ' with function from ' , sourceFile]) ;

end
end

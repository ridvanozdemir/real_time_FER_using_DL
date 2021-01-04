%% Load Training Images 
allImages = imageDatastore('f_e_r_enlarged_6_222', 'IncludeSubfolders', true,'LabelSource', 'foldernames');
%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomize'); 
%% Load Pre-trained Network (AlexNet) 
alex = alexnet;
%% Review Network Architecture 
layers = alex.Layers
%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to 
% recognize just 6 classes. 
layers(23) = fullyConnectedLayer(6); % change this based on # of classes 
layers(25) = classificationLayer
%hyperparameter tuning
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,... 
'MaxEpochs', 10, 'MiniBatchSize', 64);

trainingImages.ReadFcn = @readFunctionTrain;
%% Train the Network 

myNet = trainNetwork(trainingImages, layers, opts);
%% Test Network Performance 

testImages.ReadFcn = @readFunctionTrain; 
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

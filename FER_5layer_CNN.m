%This program is for training 5-Layer FER algorithm
%Ridvan Ozdemir

% Load training data
training_images = imageDatastore('crop_fer_dataset_1800_train', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

% Define layers
varSize = 32;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(7,'BiasLearnRateFactor',2);
fc2.Weights = gpuArray(single(randn([7 64])*0.1));

layers = [
    imageInputLayer([varSize varSize 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];

%Training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 100, ...
    'Verbose', true);

%Train
[net, info] = trainNetwork(training_images, layers, opts);

%Load test data
test_images = imageDatastore('crop_fer_dataset_1800_test', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

% Test Network Performance 
testImages.ReadFcn = @readFunctionTrain; 
predictedLabels = classify(net, test_images); 
accuracy = mean(predictedLabels == test_images.Labels)

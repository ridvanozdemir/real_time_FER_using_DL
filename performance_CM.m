%This program is for measuring performance of FER algorithm on test dataset
%Ridvan Ozdemir


%Loading test dataset
testImages = imageDatastore('crop_fer_dataset_1800_test', 'IncludeSubfolders', true,'LabelSource', 'foldernames');
testImages.countEachLabel
testImages.ReadFcn = @readFunctionTrain; 

%get predictions
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

% easy confusion matrix 
tt = table(testImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
figure; heatmap(tt,'Predicted','Actual');

%% **** Used to train SVM and MLP on HOG features **** %%
% Sources used: - https://uk.mathworks.com/help/vision/ref/extracthogfeatures.htm
%               - https://uk.mathworks.com/help/stats/classificationecoc.html
%               - https://uk.mathworks.com/help/deeplearning/ref/patternnet.html


%% Clear workspace 
clc; clear all; close all;

%% Prepare data

%Get Data
greyScaleData = '/Users/vithurshan/Desktop/FinalGREYSCALE/';
% store data in imageDatastore
imds = imageDatastore(greyScaleData,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split data for training and testing
[trainingData, testData] = splitEachLabel(imds, 0.9, 'randomized');
trainingDataSize = numel(trainingData.Files);
testDataSize = numel(testData.Files);

%% HOG features for each Image

for i = 1:trainingDataSize
    selectedImage = readimage(trainingData, i);
    trainingFeature(i, :) = extractHOGFeatures(selectedImage, 'Cellsize', [8,8]);
end
for i = 1:testDataSize
    selectedImage = readimage(testData, i);
    testFeature(i, :) = extractHOGFeatures(selectedImage, 'Cellsize', [8,8]);
end

%% SVM Classifier 

% Get Label
trainingLabels = categorical(trainingData.Labels);
testLabels = categorical(testData.Labels);

% Train model
SVMHOGClassifier = fitcecoc(trainingFeature, trainingLabels, 'FitPosterior', true);

% Generealized classification error
SVMHOGClassifierCross = crossval(SVMHOGClassifier);
genError = kfoldLoss(SVMHOGClassifierCross);
%
% Test model
[SVMHOGPredictor, score] = predict(SVMHOGClassifier, testFeature);

% Calculate accuracy
SVMHOGScore = sum(testLabels == SVMHOGPredictor)/length(testLabels)*100;
SVMHOGScore = round(SVMHOGScore,3);
strcat('Accuracy of SVM classifier with HOG features: ',  num2str(SVMHOGScore),' %')

save('SVMHOGClassifier.mat', 'SVMHOGClassifier') 

%% MLP Classifier  

% One-hot encoding
trainValue = categories(trainingLabels);
testValue = categories(testLabels);
XTrain = zeros(numel(trainValue), trainingDataSize);
XTest = zeros(numel(testValue), testDataSize);
for i = 1:trainingDataSize
    label = trainingLabels(i);
    XTrain(strcmp(trainValue, cellstr(label)),i) = 1;
end
for i = 1:testDataSize
    label = testLabels(i);
    XTest(strcmp(testValue, cellstr(label)),i) = 1;
end

% Train model
MLPHOGClassifier = patternnet(65);
MLPHOGClassifier = train(MLPHOGClassifier, trainingFeature', XTrain);
nntraintool

% Test model
MLPHOGPredictor = MLPHOGClassifier(testFeature');

% Predictions to vector
MLPHOGPredictorVector = [];
for i = 1:testDataSize
    [value, index] = max(MLPHOGPredictor(:,i));
    MLPHOGPredictorVector = [MLPHOGPredictorVector; testValue(index)];
end
MLPHOGPredictorVector = categorical(str2num(cell2mat(MLPHOGPredictorVector)));

% Calculate accuracy
MLPHOGScore = sum(testLabels == MLPHOGPredictorVector)/length(testLabels)*100;
MLPHOGScore = round(MLPHOGScore,3);
strcat('Accuracy of MLP classifier with HOG features: ',  num2str(MLPHOGScore),' %')

save('MLPHOGClassifier.mat', 'MLPHOGClassifier') 
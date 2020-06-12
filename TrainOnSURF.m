%% **** Used to train SVM and MLP on SURF features **** %%
% Sources used: - https://uk.mathworks.com/help/vision/examples/image-category-classification-using-bag-of-features.html
%               - https://uk.mathworks.com/help/vision/ref/bagoffeatures.html
%               - https://uk.mathworks.com/help/vision/ref/detectsurffeatures.html
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

%% SURF features for each Image

%create visual covabulary
bag = bagOfFeatures(imds); 

save('SURFBag.mat', 'bag') 

% Encode SURF feature vectors
trainingFeature = encode(bag, trainingData);
testFeature = encode(bag, testData);
%% SVM Classifier 

% Get Label
trainingLabels = categorical(trainingData.Labels);
testLabels = categorical(testData.Labels);

% Train and save model
SVMSURFClassifier = fitcecoc(trainingFeature, trainingLabels);
%[label,~,~,Posterior] = resubPredict(SVMSURFClassifier,'Verbose',1);

% Generealized classification error
SVMHOGClassifierCross = crossval(SVMHOGClassifier);
genError = kfoldLoss(SVMHOGClassifierCross);

% Test model
[SVMSURFPredictor, score] = predict(SVMSURFClassifier, testFeature);

% Calculate accuracy
SVMSURFScore = sum(testLabels == SVMSURFPredictor)/length(testLabels)*100;
SVMSURFScore = round(SVMSURFScore,3);
strcat('Accuracy of SVM classifier with SURF features: ',  num2str(SVMSURFScore),' %')

save('SVMSURFClassifier.mat', 'SVMSURFClassifier')
%% MLP CLassifier 

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
MLPSURFClassifier = patternnet(65);
MLPSURFClassifier = train(MLPSURFClassifier, trainingFeature', XTrain);
nntraintool

save('MLPSURFClassifier.mat', 'MLPSURFClassifier') 
%%
% Test model
MLPSURFPredictor = MLPSURFClassifier(testFeature');

% Prediction to vector
MLPSURFPredictorVector = [];
for i = 1:testDataSize
    [value, index] = max(MLPSURFPredictor(:,i));
    MLPSURFPredictorVector = [MLPSURFPredictorVector; testValue(index)];
end
MLPSURFPredictorVector = categorical(str2num(cell2mat(MLPSURFPredictorVector)));

% Calculate accuracy
MLPSURFScore = sum(testLabels == MLPSURFPredictorVector)/length(testLabels)*100;
MLPSURFScore = round(MLPSURFScore,3);
strcat('Accuracy of MLP classifier with SURF features: ',  num2str(MLPSURFScore),' %')
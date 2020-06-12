%% **** Used to train AlexNet Classifier **** %% 
% Code adopted from Lab 09 
% Sources used: - https://uk.mathworks.com/help/deeplearning/ug/transfer-learning-using-alexnet.html
%% Clear workspace 
clc; clear all; close all;

%% Prepare Data

% Get Data
RGBData = '/Users/vithurshan/Desktop/FinalRGB/';
% Store data in data store
imds = imageDatastore(RGBData, 'IncludeSubfolders',true,'LabelSource','foldernames');

% Split data for training and testing
[trainData,testData] = splitEachLabel(imds,0.9,'randomized');

%% Training

% Load network 
net = alexnet;
analyzeNetwork(net)

% Replace Final layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainData.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% optimizing the network
options = trainingOptions('sgdm', ...
 'MiniBatchSize',25, ...
 'MaxEpochs',3, ...
 'InitialLearnRate',1e-4, ...
 'LearnRateSchedule', 'piecewise', ...
 'LearnRateDropPeriod', 1, ...
 'LearnRateDropFactor', 0.1, ...
 'Shuffle','every-epoch', ...
 'ValidationData',testData, ...
 'ValidationFrequency',15, ...
 'Plots','training-progress');

% Train model
AlexNet = trainNetwork(trainData,layers,options);

%%
save('AlexNet.mat', 'AlexNet') 
%load AlexNet.mat

%% Testing

%Test model
[YPred, score] = classify(AlexNet,testData);

% Display samples
idx = randperm(numel(testData.Files),4);
figure
for i = 1:4
 subplot(2,2,i)
 I = readimage(testData,idx(i));
 imshow(I)
 label = YPred(idx(i));
 title(string(label));
end

%%
testDataLabels = testData.Labels;

% Calculate accuracy
AlexNetScore = sum(testDataLabels == YPred)/length(testDataLabels)*100;
AlexNetScore = round(AlexNetScore,3);
strcat('Accuracy of AlexNet: ',  num2str(AlexNetScore),' %')
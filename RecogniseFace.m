function P = RecogniseFace(I,featureType, classifierType, creativeMode)

% input the arguments to uppercase
featureType = upper(featureType);
classifierType = upper(classifierType);

img = I;

% fix orientation of image 
% code retrived from: https://uk.mathworks.com/matlabcentral/answers/260607-how-to-load-a-jpg-properly
image = imread(img);
info = imfinfo(img);
if isfield(info,'Orientation')
   orient = info(1).Orientation;
   switch orient
     case 1
        %normal, leave the data alone
     case 2
        image = image(:,end:-1:1,:);         %right to left
     case 3
        image = image(end:-1:1,end:-1:1,:);  %180 degree rotation
     case 4
        image = image(end:-1:1,:,:);         %bottom to top
     case 5
        image = permute(image, [2 1 3]);     %counterclockwise and upside down
     case 6
        image = rot90(image,3);              %undo 90 degree by rotating 270
     case 7
        image = rot90(image(end:-1:1,:,:));  %undo counterclockwise and left/right
     case 8
        image = rot90(image);                %undo 270 rotation by rotating 90
     otherwise
        warning(sprintf('unknown orientation %g ignored\n', orient));
   end
end

    % front face detection
faceDetection = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
faceDetection.MinSize = [50 50];
faceDetection.MergeThreshold = 6;
    % detect face and count faces
boundingBox = faceDetection(image);
facesCount = size(boundingBox, 1);
    % intialise P matrix
P = zeros(facesCount,3);
    
    % select classifier and feature type
if isequal (classifierType, 'SVM')
    
    if isequal (featureType, 'HOG')
            % load classifier
        load SVMHOGClassifier.mat;
        filteredPrediction = [];
        for i = 1:facesCount
                % crop and re-scale image
            a = boundingBox(i,:);
            scale = [120,120];
            imgGreyScale = rgb2gray(image);
            resizedImage = imresize(imcrop(imgGreyScale, a),scale);  
                % extract HOG feature 
            testFeature = extractHOGFeatures(resizedImage, 'Cellsize', [8,8]);
                % predictor
            SVMHOGPredictor = predict(SVMHOGClassifier, testFeature);
            filteredPrediction = [filteredPrediction, SVMHOGPredictor ];
                % update p matrix with label, x centre and y centre 
            P(i,1) = filteredPrediction(i);
            P(i,2) = round(boundingBox(i,1) + boundingBox(i,3)/2);
            P(i,3) = round(boundingBox(i,2) + boundingBox(i,4)/2);
        end
        
    elseif isequal (featureType, 'SURF')
            % load classifier
        load SVMSURFClassifier.mat;
        load SURFBag.mat;
        filteredPrediction = [];
        for i = 1:facesCount
                % crop and re-scale image
            a = boundingBox(i,:);
            scale = [120,120];
            imgGreyScale = rgb2gray(image);
            resizedImage = imresize(imcrop(imgGreyScale, a),scale);  
                % extract SURF feature 
            testFeature = encode(bag, resizedImage);
                % predictor
            SVMSURFPredictor = predict(SVMSURFClassifier,testFeature);
            filteredPrediction = [filteredPrediction, SVMSURFPredictor ];
                % update p matrix with label, x centre and y centre 
            P(i,1) = filteredPrediction(i);
            P(i,2) = round(boundingBox(i,1) + boundingBox(i,3)/2);
            P(i,3) = round(boundingBox(i,2) + boundingBox(i,4)/2);
        end
        
    else
       error('Please input "HOG" or "SURF" feature type for SVM classifier. %s is not valid!', featureType)
    end
    
elseif isequal (classifierType, 'MLP')
    
    if isequal (featureType, 'HOG')
            % load classifier
        load MLPHOGClassifier.mat;
        filteredPrediction = [];
        unknownCount = 101;
            % known Labels
        manualLabelling = [01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,40,42,44,46,48,50,52,54,56,58,60,78];
        for i = 1:facesCount
                % crop and re-scale image
            a = boundingBox(i,:);
            scale = [120,120];
            imgGreyScale = rgb2gray(image);
            resizedImage = imresize(imcrop(imgGreyScale, a),scale);  
                % extract HOG feature 
            testFeature = extractHOGFeatures(resizedImage, 'Cellsize', [8,8]);
                % predictor
            MLPHOGPredictor = MLPHOGClassifier(testFeature');
                % get best score
            getBestScore = max(MLPHOGPredictor); 
            transposePredictor = MLPHOGPredictor'; 
                 % get the index of best score
            predictedLabel = find(transposePredictor==getBestScore); 
                % get label 
            getLabel = manualLabelling(predictedLabel); 
            % check if face detected is unknown
            if getBestScore < 0.7 || getBestScore > 1.07
                filteredPrediction = [filteredPrediction, unknownCount];
                unknownCount = unknownCount+1;
            else
                filteredPrediction = [filteredPrediction, getLabel];
            end
                % update p matrix with label, x centre and y centre 
            P(i,1) = filteredPrediction(i);
            P(i,2) = round(boundingBox(i,1) + boundingBox(i,3)/2);
            P(i,3) = round(boundingBox(i,2) + boundingBox(i,4)/2);
        end
        
    elseif isequal (featureType, 'SURF')
            % load classifier
        load MLPSURFClassifier.mat;
            % load bag of features
        load SURFBag.mat;
        filteredPrediction = [];
        unknownCount = 101;
            % known Labels
        manualLabelling = [01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,40,42,44,46,48,50,52,54,56,58,60,78];
        for i = 1:facesCount
                % crop and re-scale image
            a = boundingBox(i,:);
            scale = [120,120];
            imgGreyScale = rgb2gray(image);
            resizedImage = imresize(imcrop(imgGreyScale, a),scale);  
                % extract SURF feature 
            testFeature = encode(bag, resizedImage);
                % predictor
            MLPSURFPredictor = MLPSURFClassifier(testFeature');
                % get best score
            getBestScore = max(MLPSURFPredictor); 
            transposePredictor = MLPSURFPredictor'; 
                 % get the index of best score
            predictedLabel = find(transposePredictor==getBestScore); 
                % get label 
            getLabel = manualLabelling(predictedLabel); 
            % check if face detected is unknown
            if getBestScore < 0.7 || getBestScore > 1.07
                filteredPrediction = [filteredPrediction, unknownCount];
                unknownCount = unknownCount+1;
            else
                filteredPrediction = [filteredPrediction, getLabel];
            end
                % update p matrix with label, x centre and y centre 
            P(i,1) = filteredPrediction(i);
            P(i,2) = round(boundingBox(i,1) + boundingBox(i,3)/2);
            P(i,3) = round(boundingBox(i,2) + boundingBox(i,4)/2);
        end
        
    else
        error('Please input "HOG" or "SURF" feature type for MLP classifier. %s is not valid!', featureType )
    end
    
elseif isequal(classifierType, 'ALEXNET')
    
    if isequal (featureType, 'NONE')
            % load classifier
        load AlexNet.mat;
        filteredPrediction = [];
            % known Labels
        manualLabelling = [01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,40,42,44,46,48,50,52,54,56,58,60,78];
        unknownCount = 101;
        for i = 1:facesCount
                % crop and re-scale image
            a = boundingBox(i,:);
            scale = [227,227];
            resizedImage = imresize(imcrop(image, a),scale);
                % predictor
            [YPred, score] = classify(AlexNet,resizedImage); 
            predictedLabel = double(string(YPred));
                % find index for prediction
            index = find(manualLabelling==predictedLabel);
            %ScoreForPrediction = score(index);
            filteredPrediction = [filteredPrediction, predictedLabel];
            
                % check if face detected is unknown
%             if ScoreForPrediction < 0.7 || ScoreForPrediction > 1.07
%                 filteredPrediction = [filteredPrediction, unknownCount];
%                 unknownCount = unknownCount+1;
%             else
%                 filteredPrediction = [filteredPrediction, predictedLabel];
%             end
                % update p matrix with label, x centre and y centre 
            P(i,1) = filteredPrediction(i);
            P(i,2) = round(boundingBox(i,1) + boundingBox(i,3)/2);
            P(i,3) = round(boundingBox(i,2) + boundingBox(i,4)/2);
        end
        
    else
        error('Please input "NONE" for AlexNet classifier. %s is not valid!', featureType )
    end
    
else
    error('Please input one of the following for classifier type: "SVM" or "MLP" or "AlexNet". %s is not valid!', classifierType)
end

if isequal (creativeMode,1)
    % Get image with prediction
    IFaces = insertObjectAnnotation(image,'rectangle',boundingBox,filteredPrediction, 'FontSize', 20);
    % Get cartoon image
    cartoonFace = 'smiley001.png';
    cartoonFace = imread(cartoonFace,'BackgroundColor',[0 0 0]);
    faceIndex = cell(facesCount,1);
    faceCartoon = cell(facesCount,1);
    % resize cartoon face according to face size
    for i = 1:facesCount
        faceBoundingBox = boundingBox(i,:);
        faceIndex{i} = imcrop(IFaces,faceBoundingBox);
        faceCartoon{i} = imresize(cartoonFace,size(faceIndex{i},1:2));
    end
    % matrix indexing for cartoon face
    for i = 1:facesCount
        row1 = boundingBox(i,1);
        row2 = boundingBox(i,2);
        col1 = boundingBox(i,3);
        col2 = boundingBox(i,4);

        IFaces(row2:row2+col2, row1:row1+col1,:) = faceCartoon{i};
        IFaces(IFaces == 0) = image(IFaces == 0);
    end
    % display results
    imshow(IFaces);
    title('Detected Faces on creative mode')
    hold on
    
elseif isequal (creativeMode,0)
    % display results
    figure;
    IFaces = insertObjectAnnotation(image,'rectangle',boundingBox,filteredPrediction, 'FontSize', 20);  
    imshow(IFaces);
    title('Detected Faces')
    hold on
else
    error('Please input 1 for creative mode and 0 for no creative mode.". %s is not valid!', creativeMode)
end

end


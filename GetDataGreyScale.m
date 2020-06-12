 %% **** Used to etract faces from Images and Videos for SVM and MLP **** %%

%%  Clear workspace 
clc; clear all; close all;

%% Root Directory
rootPath = '/Users/vithurshan/Desktop/Data/';
newPath = '/Users/vithurshan/Desktop/FinalGREYSCALE/';
%files_path = fullfile(root_path,'Individual');
subFolders = dir(rootPath);

%% Detect faces
faceDetection = vision.CascadeObjectDetector;
faceDetection.MinSize = [70 70];
faceDetection.MergeThreshold = 3;
% front face
frontFaceDetection = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
frontFaceDetection.MinSize = [70 70];
frontFaceDetection.MergeThreshold = 3;
% profile face
profileFaceDetection = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
profileFaceDetection.MinSize = [70 70];
profileFaceDetection.MergeThreshold = 3;
% resize scale
scale = [120  120];

%% Loop through files to get  images
for numberOfSubFolders = 4:size(subFolders,1)
     %sub folder name
    subFolderName = subFolders(numberOfSubFolders).name;
     %folder path
    folderPath = strcat(rootPath, subFolderName);
    imagesStorage = imageSet(folderPath, 'recursive');
    
    for i = 1:size(imagesStorage.ImageLocation,2)
        %get image
        currImage = read(imagesStorage(1), i);
        currImage = rgb2gray(currImage);
        currImage1 = imrotate(currImage,-90,'bilinear');
        %rotated anticlockwise
        currImage2 = imrotate(currImage,-85,'bilinear');
        %roated clockwise
        currImage3 = imrotate(currImage,-95,'bilinear');

        %loop through rotated and original image
        for j = 1 : 3
            selectedImage = eval(strcat('currImage',num2str(j)));
            %get bunding box using face detection 
            boundingBox = faceDetection(selectedImage);
            %if is empty move to front face detection
            if isempty(boundingBox)
                boundingBox = frontFaceDetection(selectedImage);
                if isempty(boundingBox)
                    boundingBox = profileFaceDetection(selectedImage);
                    if isempty(boundingBox)
                        continue%
                    elseif size(boundingBox,1)> 1
                        [val, idx] = max(boundingBox(:,3));
                        boundingBox =boundingBox(idx,:);
                    end
                elseif size(boundingBox,1)> 1
                    [val, idx] = max(boundingBox(:,3));
                    boundingBox =boundingBox(idx,:);
                end
            elseif size(boundingBox,1)> 1
                    [val, idx] = max(boundingBox(:,3));
                    boundingBox =boundingBox(idx,:);
            end 
                %crop image
            croppedFace = imcrop(selectedImage,boundingBox);
                %resize image and save
            resizeImage = imresize(croppedFace, scale);
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i),'_', num2str(j),'.jpg');
            imwrite(resizeImage, fileName)
                %gaussian blur
            Iblur = imgaussfilt(resizeImage,2);
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i), '_', num2str(j),'blurred','.jpg');
            imwrite(Iblur, fileName)
                %reduce brightness 
            darkerImage = resizeImage./1.5;
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i), '_', num2str(j),'darker','.jpg');
            imwrite(darkerImage, fileName)
                %increase brightness
            lighterImage = resizeImage.*1.5;
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i), '_', num2str(j),'lighter','.jpg');
            imwrite(lighterImage, fileName)
                % darker and blurred
            darkBlurImage = Iblur./1.5;
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i), '_', num2str(j),'darkBlur','.jpg');
            imwrite(darkBlurImage, fileName)
             % Kmedian filter
            KMedian = imguidedfilter(resizeImage);
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i),'_', num2str(j),'kmedian','.jpg');
            imwrite(KMedian, fileName)
                 % Kmedian filter lighter
            KMedian = imguidedfilter(resizeImage);
            lighterImage = KMedian.*1.3;
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i),'_', num2str(j),'kmedianLighter','.jpg');
            imwrite(lighterImage, fileName)
                % down sample by 1/4
            downImage = imresize(resizeImage,1/4);
            upImage = imresize(downImage, scale);
            lighterImage = upImage.*1.3;
            fileName = strcat(newPath, subFolderName, '/', 'IMG_', num2str(i),'_', num2str(j),'downscale','.jpg');
            imwrite(lighterImage, fileName)
        end 
    end 
end

%% Loop through each subfolder extract frames from all videos
 
%select only 7 frames
list = [5, 10, 15, 20, 25, 30, 35];
%list = [3, 6, 9, 12, 15, 18, 24, 27, 30, 33, 36, 39]; 

%get number of sub folders
for numberOfSubFolders = 4:size(subFolders,1)
    %sub folder name
    subFolderName = subFolders(numberOfSubFolders).name;
    %folder path
    folderPath = strcat(rootPath, subFolderName);
    %get number of videos
    videos = dir(fullfile(folderPath, '*.mp4'));
    
    for i = 1:size(videos,1)
        %get video path
        videoName = strcat(folderPath, '/',videos(i).name);
        %read video
        videoFile = VideoReader(videoName);
        %count frames
        numberOfFrames = videoFile.NumFrames;

        for frame = 1: numberOfFrames
            if ismember(frame, list)
                thisFrame = readFrame(videoFile);
                maxValue = max(thisFrame(:));
                if maxValue > 1
                    %RGB to greyscale
                    imgGreyScale = rgb2gray(thisFrame);

                    %get bunding box using face detection 
                    boundingBox = faceDetection(imgGreyScale);
                    %if is empty move to front face detection
                    if isempty(boundingBox) 
                        boundingBox = frontFaceDetection(imgGreyScale);
                        if isempty(boundingBox)
                            boundingBox = profileFaceDetection(imgGreyScale);
                            if isempty(boundingBox)
                                continue%
                            elseif size(boundingBox,1)> 1
                                [val, idx] = max(boundingBox(:,3));
                                boundingBox =boundingBox(idx,:);
                            end
                        elseif size(boundingBox,1)> 1
                            [val, idx] = max(boundingBox(:,3));
                            boundingBox =boundingBox(idx,:);
                        end
                    elseif size(boundingBox,1)> 1
                        [val, idx] = max(boundingBox(:,3));
                        boundingBox =boundingBox(idx,:);
                    end 
                    %crop image
                    croppedFace = imcrop(imgGreyScale,boundingBox);
                    %resize image
                    resizeImage = imresize(croppedFace, scale);

                    %file name for cropped frame
                    fileName = strcat(newPath, subFolderName, '/IMG_', num2str(i),'_frame', num2str(frame),'.jpg');
                    imwrite(resizeImage, fileName)
                end
            end
        end
    end
end 
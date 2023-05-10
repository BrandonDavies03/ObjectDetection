datasetFolder = "C:\Users\dcd0\OneDrive\Documents\assingment";

%digitDatasetPath = fullfile(dataset_folder,'toolbox','nnet','nndemos', ...
%    'nndatasets','DigitDataset');


imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numTrainFiles = 24;
imds.ReadFcn = @customReadDatastoreImage;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomized');
img = readimage(imdsTrain,2);
imshow(img);
reset(imds);

img = readimage(imdsTrain,2);
imshow(img);
%nnet=alexnet;

inputSize = [224 224 3];
numClasses = 2;

layers_3 = [
    imageInputLayer(inputSize)
    convolution2dLayer([5 5], 64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5],64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5], 16, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...    
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress', 'ExecutionEnvironment','gpu', ...
    InitialLearnRate=0.005,...
    MiniBatchSize=64, ...
    MaxEpochs=5);

net_3 = trainNetwork(imdsTrain,layers_3,options);


%     convolution2dLayer(5,128, Stride=2, Padding="same")
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)

%analyzeNetwork(layers);

% get the lable from the validation dataset
gt = imdsValidation.Labels;

% predict the labels given images
%pred = classify( net, imdsValidation);

% calculate the confusion matrix
%[m,order] = confusionmat(gt,pred);

% calculate the F1 and accuracy
%results = statsOfMeasure(m, 0);

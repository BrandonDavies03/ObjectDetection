
imName = '1_jpg.rf.1637bfe42fd0d0eaf1434d4ea224d54c.jpg'; %Replace with a new image so program is able to find file path
imInfo = imfinfo(imName); 

for i = 1:numel(imInfo)
    X = ['Layer ', num2str(i), ': Width ',num2str(imInfo(i).Width), ...
     ' and Height ', num2str(imInfo(i).Height)];
    disp(X)
end


net = alexnet; % load an alexnet which is pretrained 

allImages = imageDatastore('C:\Users\dcd0\OneDrive\Documents\assingment','IncludeSubfolders',true,'LabelSource','foldernames'); %Change the file path

[training_set, validation_set, testing_set] = splitEachLabel(allImages,.4,.2,.4);

layersTransfer = net.Layers(1:end-3);

categories(training_set.Labels)

numClasses = numel(categories(training_set.Labels));
imageInputSize = [224 224 3];

layers = [ %edited layers
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


lgraph = layerGraph(layers);
plot(lgraph)


augmented_training_set = augmentedImageSource(imageInputSize,training_set);

resized_validation_set = augmentedImageDatastore(imageInputSize,validation_set);
resized_testing_set = augmentedImageDatastore(imageInputSize,testing_set);

opts = trainingOptions('sgdm', ...
    'MiniBatchSize', 64,... % mini batch size
    'InitialLearnRate', 1e-0,... % fixed learning rate
    'L2Regularization', 1e-4,... % optimization L2 constraint
    'MaxEpochs',15,... % max. epochs for training, default 3
    'ExecutionEnvironment', 'gpu',...% environment for training and classification, use a compatible GPU
    'ValidationData', resized_validation_set,...
    'Plots', 'training-progress')

net = trainNetwork(augmented_training_set, lgraph, opts)

[predLabels,predScores] = classify(net, resized_testing_set, 'ExecutionEnvironment','gpu');

plotconfusion(testing_set.Labels, predLabels)
PerItemAccuracy = mean(predLabels == testing_set.Labels);
title(['overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%'])
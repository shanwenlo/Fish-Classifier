%% Fish species classification with transfer learning
clear; clc; close all;

%% 1) Load dataset
dataFolder = "fish_dataset";

imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

disp(countEachLabel(imds))

%% 2) Split data
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, "randomized");
[imdsVal, imdsTest]  = splitEachLabel(imdsTemp, 0.5, "randomized");

%% 3) Load pretrained network
net = resnet18;
inputSize = net.Layers(1).InputSize(1:2);

%% 4) Random augmentation for training data only
labelTrain = arrayDatastore(imdsTrain.Labels);
trainDS = combine(imdsTrain, labelTrain);

augTrain = transform(trainDS, @(data) randomAugmentAndResize(data, inputSize));

augVal  = augmentedImageDatastore(inputSize, imdsVal);
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% Preview augmented training images
reset(augTrain);

figure;
for i = 1:6
    data = read(augTrain);

    img = data{1};
    label = data{2};

    subplot(2,3,i);
    imshow(img);
    title(string(label));
end

reset(augTrain);

%% 5) Replace final layers
lgraph = layerGraph(net);

numClasses = numel(categories(imds.Labels));

newFCLayer = fullyConnectedLayer(numClasses, ...
    Name="new_fc", ...
    WeightLearnRateFactor=10, ...
    BiasLearnRateFactor=10);

newClassLayer = classificationLayer(Name="new_classoutput");

lgraph = replaceLayer(lgraph, "fc1000", newFCLayer);
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", newClassLayer);

%% 6) Training options
options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=2, ...
    MiniBatchSize=16, ...
    Shuffle="every-epoch", ...
    ValidationData=augVal, ...
    ValidationFrequency=10, ...
    Verbose=true, ...
    Plots="training-progress");

%% 7) Train network
trainedNet = trainNetwork(augTrain, lgraph, options);

%% 8) Evaluate on test set
YPred = classify(trainedNet, augTest);
YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest);
fprintf("Test Accuracy: %.2f%%\n", accuracy*100);

figure;
confusionchart(YTest, YPred);
title("Fish Species Confusion Matrix");

%% Local function
function dataOut = randomAugmentAndResize(dataIn, inputSize)

    img = dataIn{1};
    label = dataIn{2};

    % Resize image
    img = imresize(img, inputSize);

    %% Random 90-degree rotation (0, 90, 180, 270)
    if rand < 0.5
        k = randi([0,3]);
        img = rot90(img, k);
    end

    %% Random horizontal flip
    if rand < 0.5
        img = fliplr(img);
    end

    %% Random blur (40%)
    if rand < 0.4
        sigma = 5 * rand;
        img = imgaussfilt(img, sigma);
    end

    %% Random brightness (40%) → 0.525 to 2
    if rand < 0.4
        factor = 0.525 + (2 - 0.525) * rand;

        img = im2double(img);
        img = img * factor;
        img = min(img, 1);
        img = im2uint8(img);
    end

    dataOut = {img, label};

end
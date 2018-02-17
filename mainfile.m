%  Spam Classification using Support Vector Machines
%
% 
%The following functions are used in building the model
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%


%%——————————————Preprocessing emails——————————————————
%  To use an SVM to classify emails into Spam v.s. Non-Spam, each email need to be converted into a vector of features


% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ——————————Feature Extraction—————————————
%  Each email is converted into a vector of features in R^n

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ———————————Training Linear SVM————————————

% Load the Spam Email dataset
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% ——————————Testing Spam Classification———————————
%The classifier is evaluated both on training and test sets
% Load the test dataset
load('spamTest.mat');

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);


%——————————Top Predictors of Spam—————————————
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. These words are the most likely 
%  indicators of spam according to the classifier 
%

% Sort the weights and obtain the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');

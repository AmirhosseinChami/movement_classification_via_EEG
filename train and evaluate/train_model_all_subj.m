clc;
clear;
close all;

num_subjects = 15;
num_filters = 10;  

all_trials = cell(4, 1);  

for subj = 1 : num_subjects
    load(sprintf('../preprocess/processed_dataset/preprocessed_subj_%d.mat', subj));
    for cls = 1:4
        all_trials{cls} = cat(3, all_trials{cls}, X{cls});
    end
end

num_trials_total = size(all_trials{1}, 3);  

train_confmat = zeros(4,4);
test_confmat = zeros(4,4);

for i = 1 : 100  
    X_train = []; 
    Y_train = [];
    X_test  = []; 
    Y_test = [];

    for cls = 4 : -1 : 1
        pos_trials = all_trials{cls};
        test_trial = pos_trials(:,:,i);
        pos_trials(:,:,i) = [];
        % Data preparation for cascade CSP
        if cls == 4
            neg_trials = cat(3, all_trials{1}, all_trials{2}, all_trials{3});
            neg_trials(:,:,i) = [];  
        elseif cls == 3
            neg_trials = cat(3, all_trials{1}, all_trials{2});
            neg_trials(:,:,i) = [];
        elseif cls == 2
            neg_trials = all_trials{1};
            neg_trials(:,:,i) = [];
        else
            neg_trials = all_trials{2};
            neg_trials(:,:,i) = [];
        end

        
        pos_reshaped = reshape(pos_trials, size(pos_trials,1), []);
        neg_reshaped = reshape(neg_trials, size(neg_trials,1), []);

        [W_csp] = CSP(pos_reshaped, neg_reshaped, num_filters);

        % train features
        num_train_trials = size(pos_trials,3);
        train_features = zeros(num_train_trials, num_filters);

        for j = 1:num_train_trials
            Xf = W_csp * pos_trials(:,:,j);
            train_features(j,:) = log(var(Xf, 0, 2))';
        end
        X_train = [X_train; train_features];
        Y_train = [Y_train; cls * ones(num_train_trials,1)];

        % Extract test features
        Xf_test = W_csp * test_trial;
        test_feature = log(var(Xf_test, 0, 2))';
        X_test = [X_test; test_feature];
        Y_test = [Y_test; cls];
    end

    % For better performance. Can be commented.
    [X_train, mu, sigma] = zscore(X_train);
    X_test = (X_test - mu) ./ sigma;

    % Train cascade LDA
    model = train_cascade_LDA(X_train, Y_train);

    % Predict using cascade
    Y_train_pred = predict_cascade_LDA(model, X_train);
    Y_test_pred  = predict_cascade_LDA(model, X_test);

    train_cm = confusionmat(Y_train, Y_train_pred, 'Order', 1:4);
    test_cm  = confusionmat(Y_test,  Y_test_pred,  'Order', 1:4);

    train_confmat = train_confmat + train_cm;
    test_confmat  = test_confmat + test_cm;
end

% Accuracy
acc_test  = sum(diag(test_confmat)) / sum(test_confmat(:));
class_acc = diag(test_confmat) ./ sum(test_confmat, 2); 
acc_train = sum(diag(train_confmat)) / sum(train_confmat(:));

fprintf('Accuracy of one model for all subjects\n');
fprintf('Train Accuracy: %.2f%%\n', acc_train * 100);
fprintf('Test  Accuracy: %.2f%%\n', acc_test * 100);
for c = 1:4
    fprintf('Class %d Accuracy: %.2f%%\n', c, class_acc(c) * 100);
end

fig = figure('Name', 'All Subjects Confusion Matrices', 'NumberTitle', 'off');

% Train
subplot(1,2,1);
confusionchart(train_confmat, 1:4, 'Title', 'Train Confusion Matrix subject', ...
               'RowSummary','row-normalized', 'ColumnSummary','column-normalized');

% Test
subplot(1,2,2);
confusionchart(test_confmat, 1:4, 'Title', 'Test Confusion Matrix - All Subjects', ...
               'RowSummary','row-normalized', 'ColumnSummary','column-normalized');


saveas(fig, fullfile('../confusion matrix/cm model all subj', 'model_all_subj.fig'));
saveas(fig, fullfile('../confusion matrix/cm model all subj', 'model_all_subj.jpg'));

save('../models/one model for all subj/one_model_all_subjects.mat', 'model');

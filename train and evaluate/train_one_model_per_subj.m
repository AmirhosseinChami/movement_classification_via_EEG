clc;
clear;
close all;

num_filters_list = [14 8 14 6 14 10 14 10 14 8 14 12 10 12 6];            
num_subjects = 15;

for subj= 1:num_subjects    
    load(sprintf('../preprocess/processed_dataset/preprocessed_subj_%d.mat', subj));
    num_filters = num_filters_list(subj);

    num_trials = size(X{1}, 3); % The number of trials is the same in all classes due to preprocessing.
    train_confmat = zeros(4,4);
    test_confmat = zeros(4,4);

    for counter = 1 : num_trials * 4 
        X_train = []; 
        Y_train = [];
        X_test = [];  
        Y_test = [];

        for cls = 4 : -1 : 1
            class = X{cls};
            
            for i = 1: num_trials % LOO                
                % Data preparation for cascade CSP
                if cls == 4
                    pos_trials = class;              
                    pos_trials(:,:,i) = [];
                    
                    neg_trials = cat(3, X{1}, X{2}, X{3});
                    neg_trials(:,:,i) = [];  
                
                elseif cls == 3
                    pos_trials = class;               
                    pos_trials(:,:,i) = [];
                
                    neg_trials = cat(3, X{1}, X{2});
                    neg_trials(:,:,i) = [];
                
                elseif cls == 2
                    pos_trials = class;              
                    pos_trials(:,:,i) = [];
                
                    neg_trials = X{1};
                    neg_trials(:,:,i) = [];
                
                else
                    pos_trials = class;              
                    pos_trials(:,:,i) = [];
                
                    neg_trials = X{2};
                    neg_trials(:,:,i) = [];
                end
                
                train_trials_reshaped = reshape(pos_trials, size(pos_trials,1), []);
                ref_trials_reshaped = reshape(neg_trials,  size(neg_trials,1),  []);

                [W_csp] = CSP(train_trials_reshaped, ref_trials_reshaped, num_filters);
                
                train_trials = pos_trials;
                test_trial = class(:,:,i);
              
                % train features
                X_features = zeros(num_trials - 1, num_filters);
                labels = zeros(num_trials - 1, 1);

                for i = 1 : num_trials - 1  
                    Xi = train_trials(:, :, i);                    
                    X_train_csp = W_csp * Xi;                                                        
                    % feature = var(X_csp, 0, 2)';

                    % if i want to use log feature for better performance 
                    train_feature = log(var(X_train_csp, 0, 2))';

                    X_features(i, :) = train_feature;
                    labels(i, :) = cls;
                end
                
                X_train = [X_train; X_features];
                Y_train = [Y_train; labels];

                % Extract test features
                X_test_csp = W_csp * test_trial;
                % feature = var(X_csp, 0, 2)';

                % if i want to use log feature for better performance 
                test_feature = log(var(X_test_csp, 0, 2))';
                X_test = [X_test; test_feature];
                Y_test = [Y_test; cls];

            end
        end

        % For better performance. Can be commented.
        [X_train, mu, sigma] = zscore(X_train);
        X_test = (X_test - mu) ./ sigma;

        % Train cascade LDA
        cascade_model = train_cascade_LDA(X_train, Y_train);

        % Predict using cascade
        train_preds = predict_cascade_LDA(cascade_model, X_train);
        test_preds  = predict_cascade_LDA(cascade_model, X_test);


        train_cm = confusionmat(Y_train, train_preds, 'Order', 1:4);
        test_cm  = confusionmat(Y_test,  test_preds,  'Order', 1:4);

        train_confmat = train_confmat + train_cm;
        test_confmat  = test_confmat + test_cm;
    end

    % Accuracy
    acc_train = sum(diag(train_confmat)) / sum(train_confmat(:));
    acc_test  = sum(diag(test_confmat))  / sum(test_confmat(:));
    
    fprintf('Subject %d\n', subj);
    fprintf('Train Accuracy: %.2f%%\n', acc_train * 100);
    fprintf('Test  Accuracy: %.2f%%\n', acc_test * 100);
    
    save(fullfile('../models/one model per subj', sprintf('cascade_model_subj_%d.mat', subj)), 'cascade_model');

    fig = figure('Name', sprintf('Confusion Matrices - Subject %d', subj), 'NumberTitle', 'off');
    % Train 
    subplot(1,2,1);
    confusionchart(train_confmat, 1:4, 'Title', 'Train Confusion Matrix subject', ...
                   'RowSummary','row-normalized', 'ColumnSummary','column-normalized');

    % Test 
    subplot(1,2,2);
    confusionchart(test_confmat, 1:4, 'Title', 'Test Confusion Matrix subject ', ...
                   'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
    
    saveas(fig, fullfile('../confusion matrix/cm model per subj', sprintf('subj%d.fig',subj)));
    saveas(fig, fullfile('../confusion matrix/cm model per subj', sprintf('subj%d.jpg',subj)));
end

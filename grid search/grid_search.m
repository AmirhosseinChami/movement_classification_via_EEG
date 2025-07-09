% in this file i do a grid search to optimize the number of filters for every subject
clc;
clear;
close all;

num_subjects = 15;
filter_range = 2:2:14;      
num_eval_trials = 15;      

for subj = 1:num_subjects
    load(sprintf('../preprocess/processed_dataset/preprocessed_subj_%d.mat', subj));
    num_trials = size(X{1}, 3); 

    acc_list = zeros(length(filter_range), 1);

    for f_idx = 1:length(filter_range)
        num_filters = filter_range(f_idx);
        test_confmat = zeros(4,4);

        for counter = 1:20
            X_train = [];
            Y_train = [];
            X_test = [];
            Y_test = [];

            for cls = 4:-1:1
                class = X{cls};

                for i = 1:num_eval_trials
                    % Prepare positive and negative trials
                    pos_trials = class;
                    pos_trials(:,:,i) = [];

                    if cls == 4
                        neg_trials = cat(3, X{1}, X{2}, X{3});
                        neg_trials(:,:,i) = [];
                    elseif cls == 3
                        neg_trials = cat(3, X{1}, X{2});
                        neg_trials(:,:,i) = [];
                    elseif cls == 2
                        neg_trials = X{1};
                        neg_trials(:,:,i) = [];
                    else
                        neg_trials = X{2};
                        neg_trials(:,:,i) = [];
                    end

                    % CSP training
                    train_trials_reshaped = reshape(pos_trials, size(pos_trials,1), []);
                    ref_trials_reshaped = reshape(neg_trials, size(neg_trials,1), []);
                    cd ../'train and evaluate'/
                    [W_csp] = CSP(train_trials_reshaped, ref_trials_reshaped, num_filters);
                    
                    % Train features
                    X_features = zeros(num_eval_trials - 1, num_filters);
                    labels = zeros(num_eval_trials - 1, 1);

                    for j = 1:num_eval_trials - 1
                        Xi = pos_trials(:, :, j);
                        X_csp = W_csp * Xi;
                        feat = log(var(X_csp, 0, 2))';
                        X_features(j, :) = feat;
                        labels(j) = cls;
                    end

                    X_train = [X_train; X_features];
                    Y_train = [Y_train; labels];

                    % Test features
                    X_test_csp = W_csp * class(:,:,i);
                    test_feature = log(var(X_test_csp, 0, 2))';
                    X_test = [X_test; test_feature];
                    Y_test = [Y_test; cls];
                end
            end

            % Normalize features
            [X_train, mu, sigma] = zscore(X_train);
            X_test = (X_test - mu) ./ sigma;

            % Train and predict
            cascade_model = train_cascade_LDA(X_train, Y_train);
            test_preds = predict_cascade_LDA(cascade_model, X_test);

            test_cm = confusionmat(Y_test, test_preds, 'Order', 1:4);
            test_confmat = test_confmat + test_cm;
        end

        acc_test = sum(diag(test_confmat)) / sum(test_confmat(:));
        acc_list(f_idx) = acc_test;

        fprintf('Subject %d | Filters: %d | Accuracy: %.2f%%\n', subj, num_filters, acc_test*100);
        cd ../'grid search'/
    end

    % save the result in a CSV file
    T = table(filter_range', acc_list, 'VariableNames', {'Num_Filters', 'Accuracy'});
    csv_filename = sprintf('filter grid result/subject_%02d_results.csv', subj);
    writetable(T, csv_filename);
end

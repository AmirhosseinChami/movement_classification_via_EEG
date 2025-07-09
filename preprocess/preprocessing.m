 clc; 
clear; 
close all;

% Selected channels corresponding to the motor cortex 
selected_channels = [11,40,12,13,42,14,44,16,45,46,18,47,20,49,21,22,51,23,41,17,50];

Fs = 2400;  

% Define frequency ranges for each subject
filter_ranges = {
    [5 25], [5 25], [5 30], [10 25], [10 25], ...
    [5 30], [10 25], [8 70], [8 50], [5 30], ...
    [8 70], [8 20], [20 50], [8 70], [8 60]
};

cd ..\dataset\

num_subjects = 15;

for subj = 1:num_subjects
    X = cell(1,4);  

    temp = load(sprintf('subj_%d.mat', subj));
    data = temp.data;  

    % Design Bandpass Filter for This Subject 
    freq_range = filter_ranges{subj};
    Wn = freq_range / (Fs/2);  
    [b, a] = butter(4, Wn, 'bandpass');  

    % Find the minimum number of trials among all 4 classes
    min_trials = inf;
    for cls = 1:4
        [~, ~, trials] = size(data{cls});
        min_trials = min(min_trials, trials);
    end

    % Trim each class to the same number of trials, and select only motor cortex channels
    for cls = 1:4
        trials_data = data{cls}(selected_channels, :, 1:min_trials);

        for trial = 1:min_trials
            trials_data(:,:,trial) = filtfilt(b, a, trials_data(:,:,trial).').';
            
            % fix Nan/inf values
            temp = trials_data(:,:,trial);
            temp(isnan(temp)) = 0;
            temp(isinf(temp)) = 0;
            trials_data(:,:,trial) = temp;
        end

        X{cls} = trials_data;
    end  

    cd ..\preprocess\processed_dataset
    save(sprintf('preprocessed_subj_%d.mat', subj), 'X');
    cd ..\..\dataset\
end

disp('All subjects preprocessed with subject-specific bandpass filters.');

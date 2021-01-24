
% Read data, segmentation, filtering, periodogram PSD, 
 
% clear all; clc; 
MI_channels= [8, 9, 10, 11, ...
                13, 14, 15, ...
                18, 19, 20, 21, ...
                33, 34, 35, 36, 37, 38, 39, 40, 41];

% ####################################
%    ## (for validation data):
path= '/data/validation';
feature_label.x= [];
feature_label.y= [];
for subject=1:20 
    load(append(path,'/Data_Sample', num2str((subject)), '.mat'));
    y= Validation.y_dec - 1;
    for trial=1:20
            tmp(:,:)= filter(myfilter, Validation.x(:, trial, MI_channels));  
        x(:, :)= tmp[1001:3500, :]
        [pxx, w]= periodogram(x);
        pxx_trial(trial, :)= reshape(pxx, [size(pxx,1)*size(pxx,2), 1]);
    end
    feature_label.x= [feature_label.x; pxx_trial];
    feature_label.y= [feature_label.y; y'];
    clear x pxx_trial y 
end
save('feature_label_val.mat', 'feature_label')

% ####################################
%    ## (Repeat for training data): 
clear feature_label
path= '/data/train';
feature_label.x= [];
feature_label.y= [];
for subject=1:20 
    load(append(path,'/Data_Sample', num2str((subject)), '.mat'));
    y= Training.y_dec - 1;
    for trial=1:20
            tmp(:,:)= filter(myfilter, Training.x(:, trial, MI_channels));  
        x(:, :)= tmp[1001:3500, :]
        [pxx, w]= periodogram(x);
        pxx_trial(trial, :)= reshape(pxx, [size(pxx,1)*size(pxx,2), 1]);
    end
    feature_label.x= [feature_label.x; pxx_trial];
    feature_label.y= [feature_label.y; y'];
    clear x pxx_trial y 
end

save('feature_label_train.mat', 'feature_label')

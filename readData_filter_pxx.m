MI_channels= [8   9  10  11  13  14  15  18  19  20  21  33  34  35  36  37  38  39  40  41];

%% Training
clear
% ch_l= [9, 33, 8, 36, 13, 35, 19, 39, 18, 42, 24, 61, 12, 47, 48, 49, 23, 29];
% ch_r= [10, 34, 11, 37, 15, 38, 20, 41, 21, 43, 26, 62, 16, 52, 53, 54, 27, 31];
ch_l= [ 36, 13, 35, 19, 39, 18];
ch_r= [37, 15, 38, 20, 41, 21];
ch_all= [ch_l, ch_r];
path= '/Users/Parisa/Google Drive/Side-projects/EEG-BCI/NER2021/data';

% for indx= 1: length(ch_l)
%     if ismember(indx, [4, 5])
%     append('tr',num2str(indx))
    x_all= [];
    y_all= [];
    for i=1:1
        load(append(path,'/validation/Data_Sample', num2str((i)), '.mat'));

        y_= Validation.y_dec - 1;
        segments= {[1001:1500], [1801:2300], [2601:3100]};
        x_trial=[];
        y_trial=[];
        for trial=1:1
%             for ch_indx = 1:length(ch_all)
%                 tmp_l(:,:)= filter(myfilter, Training.x(:, trial, ch_l(indx)));
%                 tmp_r(:,:)= filter(myfilter, Training.x(:, trial, ch_r(indx)));

                tmp_all(:,:)= filter(myfilter, Validation.x(:, trial, ch_all));

%             end  
%             x(:, :)= [tmp_l(1101:3100, :) , tmp_r(1101:3100, :)];
            for s = 1: length(segments)
                x(:, :)= tmp(segments{s}, :); 
                [pxx, w]= periodogram(x);
                x_s= reshape(pxx, [size(pxx,1)*size(pxx,2), 1]);
%                 x_s(s, :)= reshape(x, [size(x,1)*size(x,2), 1]);
            end
            y_s= [y_(trial), y_(trial), y_(trial)];
            x_trial= [x_trial; x_s];
            y_trial= [y_trial, y_s];
        end
        x_all= [x_all; x_trial];
        y_all= [y_all; y_trial'];
        
%         clear x x_trial y 

%         csvwrite(append(path, '/channel_psd_perSubj/y_ch_psd_tr2' , num2str(indx), 'subj', num2str(i), '.csv'), y_all)
    end
%     csvwrite(append(path, '/segmented_psd/xval.csv'), x_all)
%     csvwrite(append(path, '/segmented_psd/yval.csv'), y_all)
% %     
%         csvwrite(append(path, '/channel_psd_perSubj/x_ch_psd_tr2', num2str(indx), 'subj', num2str(i), '.csv'), x_all)
%         csvwrite(append(path, '/channel_psd_perSubj/y_ch_psd_tr2' , num2str(indx), 'subj', num2str(i), '.csv'), y_all)
%     clear x_all y_all
%     end
% end




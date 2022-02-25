
clc; clear all
% discard the channels that are nor related to the motor imagery functions. (see the data set for information about the channels)
channels= [9, 10, 33, 34, 8, 11, 36, 37, 13, 15, ...
            35, 38, 14, 19, 20, 39, 41, 18, 21, 40];
feature_label.x= [];
feature_label.y= [];

for i = 1:20 %subjects x_1 to x_20
    a= append('Tr', num2str((i)), '.mat');
    dat= load(append('filtered data/Training set/', a));
    clear a
    y = zeros(20, 1);
    for ii = 1:20 % trials x_1 to x_20
        x_tmp0= dat.(append('x_', num2str(ii)))(:, channels);
%         x_tmp= myfilter(x_tmp);
        for jj = 1:size(x_tmp0,2)
           x_tmp(:, jj)= filter(myfilter, x_tmp0(:,jj)); 
        end
        y(ii)= dat.(append('y_', num2str(ii)));
        [pxx, w]= periodogram(x_tmp);
%         pxx= pxx(1:120,:);
%         figure; bar(pxx(1:150, :),'DisplayName','pxx')   
%         while (10*j+10) < size(pxx,1)
%         for j = 0:7
%             pxx_w1(j+1,:)= max(pxx(5*j+1: 5*j+5, :));
%         end
%         for jj= 4:10
%             pxx_w2(jj-3,:)= max(pxx(10*jj+1 :10*jj+10 ,:));
%         end
        
%         pxx_window= [pxx_w1; pxx_w2];
        x(ii,:)=reshape((pxx), [size(pxx,1)*size(pxx,2), 1]);

%         figure; bar(pxx_window,'DisplayName','pxx_window') %plotmatrix(pxx_window)
%         x(ii,:)=reshape(pxx_window, [size(pxx_window,1)*size(pxx_window,2), 1]);
    end
    feature_label.x= [feature_label.x; x];
    feature_label.y= [feature_label.y; y-1];
%     clear x y x_tmp dat
end
feature_label.x= feature_label.x(2:end, :);
feature_label.y= feature_label.y(2:end, :);

save('feature_label_train_5-15.mat', 'feature_label')

% fl= load('feature_label_val_5-15.mat');

csvwrite('x_pxxfilt_tr.csv', feature_label.x)
csvwrite('y_pxxfilt_tr.csv', feature_label.y)


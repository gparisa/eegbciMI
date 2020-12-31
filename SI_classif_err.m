clc; clear; 
xy_train= load('feature_label_train.mat');
xy_val= load('feature_label_val.mat');
dat.x= [xy_train.feature_label.x ; xy_val.feature_label.x];
dat.y= [xy_train.feature_label.y ; xy_val.feature_label.y];

g0 = dat.x(dat.y==0,:);
g1 = dat.x(dat.y==1,:);
[h,p,ci,stat] = ttest2(g0, g1, 'Vartype', 'unequal');
[~,featureIdxSortbyP] = sort(p,2); % sort the features
p_sort= sort(p);
num= sum(p_sort<0.005);
dat.x = dat.x(:,featureIdxSortbyP(1:num)); 

n_test= 380;
rng(14); % fixed for reproducibility 
test_index= randsample(size(dat.x,1), n_test);
test_dat.x= dat.x(test_index, :); 
test_dat.y= dat.y(test_index, :);

train_dat= dat;
train_dat.x(test_index, :)=[];
train_dat.y(test_index, :)=[];

c=[100, 150, 200, 250, 300, 350, 400];
reps= 100;
for i = 1:reps
    for ii = 1:length(c)
        indx= randsample(size(train_dat.x,1), c(ii));
        mdllda = fitcdiscr(dat.x(indx,:), dat.y(indx));
        hat_y= mdllda.predict(test_dat.x);
        err_lda(i,ii)= mean(hat_y == test_dat.y);
        clear hat_y
        
        mdlsvm = fitcsvm(dat.x(indx,:), dat.y(indx), 'KernelFunction', 'rbf');
        hat_y= mdlsvm.predict(test_dat.x);
        err_svmrbf(i,ii)= mean(hat_y == test_dat.y);
        clear hat_y
        
        mdlcart3 = fitctree(dat.x(indx,:), dat.y(indx), 'MinLeafSize', 2);
        hat_y= mdlcart3.predict(test_dat.x);
        err_cart3(i,ii)= mean(hat_y == test_dat.y);
        clear hat_y

        mdl3nn = fitcknn(dat.x(indx,:), dat.y(indx), 'NumNeighbors', 3);
        hat_y= mdl3nn.predict(test_dat.x);
        err_3nn(i,ii)= mean(hat_y == test_dat.y);
        clear hat_y

    end
end

%% ----- Plots -----
figure; boxplot(100*err_lda); title('LDA'); 
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

figure; boxplot(100*err_svmrbf); title('SVM')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

figure; boxplot(100*err_3nn); title('3NN')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

figure; boxplot(100*err_cart3, 'fontSize', 18); title('CART-3')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'}) 

t= 1:length(c);
figure; plot(t,mean(err_lda), ...
     t, mean(err_svmrbf), ...
    t, mean(err_cart3),  ...
    t, mean(err_3nn), 'linewidth', 2); 
legend('LDA', 'SVM', 'CART', '3NN') 
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

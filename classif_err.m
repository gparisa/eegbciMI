clc; clear; 
xy_train= load('feature_label_train.mat');
xy_val= load('feature_label_val.mat');
dat.x= [xy_train.feature_label.x ; xy_val.feature_label.x];
dat.y= [xy_train.feature_label.y ; xy_val.feature_label.y];

g0 = dat.x(dat.y==0,:);
g1 = dat.x(dat.y==1,:);
[h,p,ci,stat] = ttest2(g0, g1, 'Vartype', 'unequal');
% ecdf(p);xlabel('P value'); ylabel('CDF value')

[~,featureIdxSortbyP] = sort(p,2); % sort the features
p_sort= sort(p);
num= sum(p_sort<0.005);
dat.x = dat.x(:,featureIdxSortbyP(1:num)); 

rng(14); %rng('default'); 

n_test= 380;
test_index= randsample(size(dat.x,1), n_test);
test_dat.x= dat.x(test_index, :); 
test_dat.y= dat.y(test_index, :);

train_dat= dat;
train_dat.x(test_index, :)=[];
train_dat.y(test_index, :)=[];

c=[100, 150, 200, 250, 300, 350, 400];
reps= 50;
for i =1:reps
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
        
%         mdlcart5 = fitctree(dat.x(indx,:), dat.y(indx), 'MinLeafSize', 10);
%         hat_y= mdlcart5.predict(test_dat.x);
%         err_cart5(i,ii)= mean(hat_y == test_dat.y);
%         clear hat_y

        mdl3nn = fitcknn(dat.x(indx,:), dat.y(indx), 'NumNeighbors', 3);
        hat_y= mdl3nn.predict(test_dat.x);
        err_3nn(i,ii)= mean(hat_y == test_dat.y);
        clear hat_y

%         mdl5nn = fitcknn(dat.x(indx,:), dat.y(indx), 'NumNeighbors', 5);
%         hat_y= mdl5nn.predict(test_dat.x);
%         err_5nn(i,ii)= mean(hat_y == test_dat.y);
%         clear hat_y

    end
end

figure; boxplot(100*err_lda); title('LDA'); 
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

figure; boxplot(100*err_svmrbf); title('SVM')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})

figure; boxplot(100*err_3nn); title('3NN')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})
% figure; boxplot(err_5nn); title('5NN')

figure; boxplot(100*err_cart3, 'fontSize', 18); title('CART-3')
ylim([50 85])
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})
% figure; boxplot(err_cart5); title('CART-10')

 
rms_lda= sqrt(mean(err_lda).^2 + var(err_lda));
rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
rms_cart3= sqrt(mean(err_cart3).^2 + var(err_cart3));
rms_cart5= sqrt(mean(err_cart5).^2 + var(err_cart5)); 
rms_3nn= sqrt(mean(err_3nn).^2 + var(err_3nn)); 
rms_5nn= sqrt(mean(err_5nn).^2 + var(err_5nn));
% 
%
t= 1:length(c);
figure; plot(t,mean(err_lda), ...
     t, mean(err_svmrbf), ...
    t, mean(err_cart3),  ...
    t, mean(err_3nn), 'linewidth', 2); 
legend('LDA', 'RBF SVM', 'CART-MLS3', '3NN') 
title('Bias'); 
grid on; xticklabels({'100', '150', '200', '250', '300', '350', '400'})


aa= [mean(err_lda)', mean(err_svmrbf)', mean(err_cart3)', mean(err_3nn)'];
aa= fix(round(aa,2)*100)

bb= [std(err_lda)', std(err_svmrbf)', std(err_cart3)', std(err_3nn)'];
bb= fix(round(bb,2)*100)
% % %%
% % t= 1:c;
% % figure; plot(t,var(err_lda), ...
% %      t, var(err_svmrbf), ...
% %     t, var(err_cart3), t, var(err_cart5), ...
% %     t, var(err_3nn), t, var(err_5nn)); 
% % legend('LDA', 'RBF SVM', 'CART-MLS3', 'CART-MLS10', '3NN', '5NN') 
% % title('Variance')
% % grid on 
% % %%
t=1:length(c);
figure; plot(t,rms_lda, ...
     t, rms_svmrbf, ...
    t, rms_cart3, t, rms_cart5, ...
    t, rms_3nn, t, rms_5nn); 
legend('LDA', 'RBF SVM', 'CART-MLS3', 'CART-MLS10', '3NN', '5NN') 
title('RMS')
grid on         

% %% 
% 
% % n50_errors= [err_lda(:,1), err_svmrbf(:,1), err_3nn(:,1), err_5nn(:,1), err_cart(:,1)]
% 
% for i=1:5
%     figure;
%     boxplot([err_lda(:,i), err_svmrbf(:,i), ...
%         err_3nn(:,i), err_5nn(:,i), ...
%         err_cart3(:,i)])
%     title(['n = ',num2str(50*i)])
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %%
% 
% 
% 
% 
% % figure;plot(rms_lda, 'marker', 'o')
%  
% 
% % for i =1:reps
% %     for ii = 1:7
% %         indx= randsample(size(train_dat.x,1), ii*50);
% %         mdlsvm = fitcsvm(dat.x(indx,:), dat.y(indx), 'KernelFunction', 'linear');
% %         hat_y= mdlsvm.predict(test_dat.x);
% %         err_svm(i,ii)= mean(hat_y ~= test_dat.y);
% %     end
% % end
% % figure; boxplot(err_svm); title('Linear SVM')
% %  
% % rms_svm_linear= sqrt(mean(err_svm).^2 + var(err_svm));
% % % figure;plot(rms_svm, 'marker', 'o', 'title', 'Linear SVM')
% 
% 
% for i =1:reps
%     for ii = 1:c
%         indx= randsample(size(train_dat.x,1), ii*50);
%         mdlsvm = fitcsvm(dat.x(indx,:), dat.y(indx), 'KernelFunction', 'rbf');
%         hat_y= mdlsvm.predict(test_dat.x);
%         err_svmrbf(i,ii)= mean(hat_y ~= test_dat.y);
%     end
% end
% % figure; boxplot(err_svmrbf); title('RBF SVM')
%  
% rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
% % figure;plot(rms_svmrbf, 'marker', 'o'); title ('RBF-SVM')
% 
% 
% for i =1:reps
%     for ii = 1:c
%         indx= randsample(size(train_dat.x,1), ii*50);
%         mdlcart3 = fitctree(dat.x(indx,:), dat.y(indx), 'MinLeafSize', 2);
%         hat_y= mdlcart3.predict(test_dat.x);
%         err_cart3(i,ii)= mean(hat_y ~= test_dat.y);
%     end
% end
% rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
% rms_cart3= sqrt(mean(err_cart3).^2 + var(err_cart3));
% 
% for i =1:reps
%     for ii = 1:c
%         indx= randsample(size(train_dat.x,1), ii*50);
%         mdlcart5 = fitctree(dat.x(indx,:), dat.y(indx), 'MinLeafSize', 10);
%         hat_y= mdlcart5.predict(test_dat.x);
%         err_cart5(i,ii)= mean(hat_y ~= test_dat.y);
%     end
% end
% % figure; boxplot(err_cart); title('CART')
% rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
% rms_cart3= sqrt(mean(err_cart3).^2 + var(err_cart3));
% rms_cart5= sqrt(mean(err_cart5).^2 + var(err_cart5));
% % figure;plot(rms_cart, 'marker', 'o', 'title', 'CART')
% 
% for i =1:reps
%     for ii = 1:c
%         indx= randsample(size(train_dat.x,1), ii*50);
%         mdl3nn = fitcknn(dat.x(indx,:), dat.y(indx), 'NumNeighbors', 3);
%         hat_y= mdl3nn.predict(test_dat.x);
%         err_3nn(i,ii)= mean(hat_y ~= test_dat.y);
%     end
% end
% % figure; boxplot(err_3nn); title('3NN')
% rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
% rms_cart3= sqrt(mean(err_cart3).^2 + var(err_cart3));
% rms_cart5= sqrt(mean(err_cart5).^2 + var(err_cart5)); 
% rms_3nn= sqrt(mean(err_3nn).^2 + var(err_3nn));
% 
% %%5nn
% for i =1:reps
%     for ii = 1:c
%         indx= randsample(size(train_dat.x,1), ii*50);
%         mdl5nn = fitcknn(dat.x(indx,:), dat.y(indx), 'NumNeighbors', 5);
%         hat_y= mdl5nn.predict(test_dat.x);
%         err_5nn(i,ii)= mean(hat_y ~= test_dat.y);
%     end
% end
% % figure; boxplot(err_5nn); title('5NN')
% 
% 
% rms_svmrbf= sqrt(mean(err_svmrbf).^2 + var(err_svmrbf));
% rms_cart3= sqrt(mean(err_cart3).^2 + var(err_cart3));
% rms_cart5= sqrt(mean(err_cart5).^2 + var(err_cart5)); 
% rms_3nn= sqrt(mean(err_3nn).^2 + var(err_3nn)); 
% rms_5nn= sqrt(mean(err_5nn).^2 + var(err_5nn));
% 
% t= 1:c;
% figure; plot(t,rms_lda, ...
%      t, rms_svmrbf, ...
%     t, rms_cart3, t, rms_cart5, ...
%     t, rms_3nn, t, rms_5nn); 
% legend('LDA', 'RBF SVM', 'CART-MLS3', 'CART-MLS10', '3NN', '5NN') 
% grid on         
% 
% %% 
% 
% % n50_errors= [err_lda(:,1), err_svmrbf(:,1), err_3nn(:,1), err_5nn(:,1), err_cart(:,1)]
% 
% for i=1:5
%     figure;
%     boxplot([err_lda(:,i), err_svmrbf(:,i), ...
%         err_3nn(:,i), err_5nn(:,i), ...
%         err_cart3(:,i), err_cart5(:,i)])
%     title('n', num2str(i))
% end
% 
% 
% 

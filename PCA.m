addpath(genpath('.'));

%% Loading step
% loading of the files, mfeat-pix contains 200*240
load mfeat-pix.txt -ascii;
%%
% reconstuction percent which can be manually be changed
rcons = 1;  

% taking training samples of all '3'
xtrain = mfeat_pix(601:800, 1:240);

% calculating the centered x by subtracting with mean
xcentered = zeros(200, 240);
% for i = 1:200
%     xcentered(i, :) = xtrain(i, :) - mean(xtrain(i, :));
% end
mean = sum(xtrain(:,:)) / 200;
xcentered = xtrain - repmat(mean, 200, 1);

% covariance martrix formation
xCov = (1/200)*(xcentered' * xcentered);

% Applying SVD on the covariance matrix
% U is the feature matrix, S is the eigen value matrix and V is the weight
% matrix
[U, S, V] = svd(xCov);
%%

% the sum of all eigen values
den = sum(diag(S.^2));
% k will hold the value for with the feature vectors ca be taken
k = 1;
num = 0;
% checking if the ratio 
while(k<240)
    for i = 1:k
        num = num + (S(i,i).^2);
    end
    if ((num/den)> rcons)
        break;
    end
    num = 0;
    k = k + 1;
end
k

projection = ((U(:,1:k))' *  xcentered')';
xfinal = projection * U(:,1:k)';

xfinal = xfinal + repmat(mean, 200, 1);
%%
%plot the figure  

figure(1);
for i = 1:5
    for j = 1:5
        pic = xfinal(1 * (i-1)+j ,:);  
        picmatreverse = zeros(15,16);
        picmatreverse(:)= - pic;
        picmat = zeros(15,16);
        for k = 1:15
            picmat(:,k)=picmatreverse(:,16-k);
        end
        subplot(5,5,(i-1)* 5 + j);
        pcolor(picmat');
        axis off;
        colormap(gray(10));
    end
end
addpath(genpath('.'));

%% Loading step
% loading of the files, mfeat-pix contains 2000*240
load mfeat-pix.txt -ascii;
%%
% train is 1000*240 matrix with 100 of each of digits
% test is 1000*240 matrix with 100 of each of digits
N = 1000;
train = mfeat_pix(1:100, 1:240);
test = mfeat_pix(101:200, 1:240);
% mechanism to get the required 1000 and 1000 test and train data uniformly
% for all the digits
for i = 1:9
    train = [train; mfeat_pix((200*i)+1:(200*i)+100, 1:240)];
    test = [test; mfeat_pix((200*i)+101:(200*i)+200, 1:240)];
end
%%
% Making the Z matrix which is composed of binary numbers depecting the
% position of the digits. It is 1000*10 matrix
Zmat = [];
for i = 0:9
    for j = 1:100
        bin = de2bi((2^i),10);
        Zmat = [Zmat; bin];
    end
end
%%
% calculating the centered x by subtracting with mean
mean_train = sum(train(:,:)) / N;
xcentered_train = train - repmat(mean_train, N, 1);

mean_test = sum(test(:,:)) / N;
xcentered_test = test - repmat(mean_test, N, 1);

% covariance martrix formation
xCov_train = (1/N)*(xcentered_train' * xcentered_train);
xCov_test = (1/N)*(xcentered_test' * xcentered_test);

% Applying SVD on the covariance matrix
% U is the feature matrix, S is the eigen value matrix and V is the weight
% matrix
[U_train, S_train, V_train] = svd(xCov_train);

%%
% these vectors hold the final results... number of misclassification
x_points = [];
y_test_points = [];
y_train_points = [];

for i = 1:48
    x_points = [x_points, i*5];
    m = x_points(i);  % selection of m manually
    % selecting the m PCs for both train and test data
    U_mtrain = U_train(:,1:m);
    % making m feature matrix from both test and train data
    f_mtrain = (U_mtrain' * xcentered_train');
    f_mtest = (U_mtrain'* xcentered_test');
    % the phi matrix is formed from feature matrix and is of 1000*m
    phi = f_mtrain';
    %Finding optimal weight matrix
    Wopt = (pinv(phi)* Zmat);
    % finding the resultant Z matrix of test and train
    Ztrain = (Wopt' * f_mtrain)';
    Ztest = (Wopt' * f_mtest)';
    % taking the column wise result
    Z_test = Ztest';
    Z_train = Ztrain';
    Z_mat = Zmat';
    % finding out the maximum weightage at the given index
    [Mtest, I_test] = max(Z_test);
    [Mtrain, I_train] = max(Z_train);
    [M, Index] = max(Z_mat);
    count1 = 0;
    count2 = 0;
    % counting the testing misclassification
    for j = 1:1000
        if I_test(j) ~= Index(j)
            count1 = count1 + 1;
        end
    end
    % counting the training misclassification
    for a = 1:1000
        if I_train(a) ~= Index(a)
            count2 = count2 + 1;
        end
    end
    % out of total number of the vector we find the fraction
    count1 = count1/1000;
    count2 = count2/1000;
    % collection of the points 
    y_test_points  = [y_test_points, count1];
    y_train_points  = [y_train_points, count2];
end

%%
 y_test_points = log10(y_test_points);
 y_train_points = log10(y_train_points);
 
% plotting the final result
figure
plot(x_points, y_test_points, '-r')
hold on
plot(x_points, y_train_points, '-b')

title('plot of training and testing misclassification')
xlabel('number of PCs')
ylabel('misclassification')



        


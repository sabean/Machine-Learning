addpath(genpath('.'));

%% Loading/reading step
% loading of the files, mfeat-pix contains 240*2000
load mfeat-pix.txt -ascii;

%% Spliting data sets into train and test
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
folds = 10;
[model_set] = create_model();
[average, best_model_index] = cross_validation_loop(train, model_set, folds);
best_model = model_set(best_model_index, :)



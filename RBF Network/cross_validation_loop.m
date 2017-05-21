function [average_miss, optimal_index] = cross_validation_loop(train, model_tensor, folds)
    % Preliminary for the cross validation
    % correct labels matrix for validation set
    b = ones(1,100);
    Zmat = blkdiag(b, b, b, b, b, b, b, b, b, b)';
    numModel = size(model_tensor, 1);
    fold_sets = cvpartition(size(train, 1), 'KFold', folds);
    for i = 1:folds
        fprintf('fold = %d\n', i);
        % training set and index for true position
        train_set = train(training(fold_sets, i), :);
        Zmat_training = Zmat(training(fold_sets, i), :);
        % validation set and index for true position
        validation_set = train(test(fold_sets, i), :);
        Zmat_validation = Zmat(test(fold_sets, i), :);
        % training and validation set
        for j = 1:numModel
            fprintf('numModel = %d\n', j);
            alpha = model_tensor(j,2);
            beta = model_tensor(j, 3);
            noRbfeatures = model_tensor(j,4);
            % RBF feature extraction method 
            [ rbf_features, centroid ] = RBFfeatures(train_set, noRbfeatures, beta);
            % Optimal set by ridge regression method
            m = size(rbf_features,2);
            w_opt = ((inv(rbf_features' * rbf_features + alpha^2 * eye(m))) * rbf_features' * Zmat_training)';
            % testing the values in the validation set
            [misclassification_rates] = data_test_mechanism(w_opt, validation_set, Zmat_validation, centroid, beta);
            misclassification_list(j) = misclassification_rates;
        end
        total_missed_rates(i, :) = misclassification_list;
    end
    average_miss = mean(total_missed_rates);
    minimum = min(total_missed_rates);
    optimal_index = find(average_miss == minimum);
end

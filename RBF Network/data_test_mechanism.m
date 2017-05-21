function [misclassification_rates] = data_test_mechanism(optimalset, trainset, Zlabels, centroid, beta)
    [test_features] = Radial_basis_function(trainset, centroid, beta);
    test_outcome_set = optimalset * test_features';
    [~ , tindex] = max(test_outcome_set);
    [~ , correctIndex] = max(Zlabels, [], 2);
    misclassification_rates = sum(correctIndex' ~= tindex)/ size(trainset,1);
end

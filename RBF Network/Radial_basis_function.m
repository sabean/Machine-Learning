function [rbf_features] = Radial_basis_function(train, centroid, beta)
    rbf_features = zeros(size(train, 1), size(centroid, 1));
    for i = 1:size(train, 1)
        for j = 1:size(centroid, 1)
            rbf_features(i,j) = exp(-(norm( train(i,:) - centroid(j,:) )^2 / beta));
        end
    end
end
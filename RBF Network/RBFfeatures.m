function [ rbf_features, centroid ] = RBFfeatures(train, noRbfeatures, beta)
    [~, centroid] = kmeans(train, noRbfeatures);
    rbf_features = Radial_basis_function(train, centroid, beta);
end

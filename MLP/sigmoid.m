function [value] = sigmoid(in)
    value = 1./(1 + exp(-in));
end
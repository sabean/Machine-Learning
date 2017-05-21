clc
clear all
x = [[0,0];
     [0,1];
     [1,0];
     [1,1]];
y =[0; 1; 1; 0];

lambda = 0.7;

% MSE declaration
epoch = 20000;
mse = zeros(1, epoch); 

% Weight initializations
W1x = randn(2,2);
W2x = randn(1,2);
b1 = randn(2,1);
b2 = randn(1,1);
% W1x = [-1.2646 -1.3122; 4.0617 4.4264];
% W2x = [3.8341 2.8150];
% b1 = [1.4341 ;-1.2157];
% b2 = -3.7398;

ground = zeros(1,size(x, 1));
theta0 = [b1(1,1); W1x(1,1); W1x(1,2); b1(2,1); W1x(2,1); W1x(2,2); b2; W2x(1,1);W2x(1,2)];
for j = 1:epoch
    gradient = zeros(9,1);
    W1x = [theta0(2) theta0(3); theta0(5) theta0(6)];
    W2x = [theta0(8) theta0(9)];
    b1 = [theta0(1) ; theta0(4)];
    b2 = theta0(7);
    fprintf('------- epoch %d ------\n\n', j);
    for i = 1:4
        x0 =[x(i,1); x(i,2)];  % real input layer
        yt = y(i);  % real output layer
        
        % forward pass
        
        x1 = sigmoid(W1x*x0 + b1);   % Hidden layer
        x2 = W2x*x1 + b2;  % estimated output layer
        fprintf('input:  %d   %d\n', x0(1), x0(2));
        fprintf('       Estimated output:  %f\n\n', x2);
        
        % backward pass
        
        del21 = 2*(x2 - yt);
        del11 = (x1(1)*(1-x1(1)))*(del21*W2x(1,1));
        del12 = (x1(2)*(1-x1(2)))*(del21*W2x(1,2));
        
        % gradient
        gradient = gradient + [del11; del11*x0(1); del11*x0(2); del12; del12*x0(1); del12*x0(2); del21; del21*x1(1); del21*x1(2)];
        
        % evaluate the output
        if (x2 > 0.5)
            y_mlp = 1;
        else
            y_mlp = 0;
        end
        ground(i) = abs(y_mlp-x2);
    end
    mse(j) = mean(ground);
    ground = zeros(1,size(x, 1));
    fprintf('\n\n');
    gradient = gradient./4;
    theta0 = theta0 - lambda*gradient;
end
%% draw the MSE against number of epochs

e = 1:epoch;
figure(1)
p = plot(e, mse);
p(1).LineWidth = 2;
legend('show');
xlabel('Epochs');
ylabel('Misclassification rate');



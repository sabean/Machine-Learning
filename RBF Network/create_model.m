function [model_set] = create_model(~)
    % alpha = linspace(0, 1, 23);
    % beta = linspace (1, 20, 23);
    % clusters = linspace(240, 350, 23);
    % model_num = linspace(1, 23, 23);

    % clusters = linspace(580, 650, 15);
    % alpha = linspace(0, 0.14, 15);
    % beta = linspace(1000, 2400, 15);
    % model_num = linspace(1, 15, 15);
    clusters = 600;
    alpha = 0.01;
    beta = 1000;
    model_num = 1;
    model_set = [model_num', alpha', beta', clusters'];
end
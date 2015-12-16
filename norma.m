function [precision] = norma(x, y, kernel_sigma, rho, nu, do_truncation, tau)
    addpath('./helper/');
    addpath('./data/handwritten_for_classification');
%     load('data/mnist_49_3000.mat');
%     load('data/mnist_binary.mat');

%     [x, y] = get_handwritten(1, 2);
    
%     [x, y] = gen_data();
    
%     x = x';
%     y = y';
%     size_choose = 1000;
%     x = x(1:size_choose, :);
%     y = y(1:size_choose, :);

%     do_truncation = 0; % indicate whether to do truncation
%     tau = 200; % the number of data to "remember"

    eta = 0.01;
%     rho = 0;
    lambda = 1;
%     nu = 0.01;
%     kernel_sigma = 4;
    b = 0;
    n = size(x, 1);

    alphas = [];

    t = 1;
    correct = 0;
    while t <= n
        eta = 0.0001 / sqrt(t);
        if t == 1
            g_x = 1;
%             alphas = [alphas, -eta * loss_gradient_sm(g_x, y(t, :), rho)];
            [new_alpha, b, rho] = update_alpha_b(g_x, y(t, :), eta, rho, b, nu);
            alphas = [alphas, new_alpha];
        else
            % If do_truncation, need to truncate k_mat to 
            if do_truncation
                x_taus = x(max(1, t - tau): t - 1, :);
                k_mat = kernel_gaussian(x_taus, x(t, :), kernel_sigma);
            else
                k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
            end
            g_x = alphas * k_mat + b;
            alphas = (1 - eta * lambda) * alphas;
            [new_alpha, b, rho] = update_alpha_b(g_x, y(t, :), eta, rho, b, nu);
            if do_truncation && t - tau >= 1
                alphas = [alphas(:, 2:end), new_alpha];
            else
                alphas = [alphas, new_alpha];
            end
        end
        if g_x * y(t, :) > 0
            correct = correct + 1;
        end
%         fprintf('%d %d %f', t, correct, correct / t);
%         disp([b, rho, eta]);
        precision = correct / t;
        t = t + 1;
    end
end

function [ alpha, new_b, new_rho ] = update_alpha_b( f_x, y, eta, rho, b, nu )
    if f_x * y <= rho
        alpha = eta * y;
        new_b = b + eta * y;
        new_b = b;
%         new_rho = rho + eta * (nu - 1);
        new_rho = rho;
    else
        alpha = 0;
        new_b = b;
%         new_rho = rho + eta * nu;
        new_rho = rho;
    end
end

function [x, y] = gen_data()
    total_range =10000;
    x = zeros(total_range,2);
    y = zeros(total_range, 1);
    umin=-1;
    umax=1;
    n=total_range;
    u=umin+rand(1,n)*(umax-umin);
    y = sign(u)';
    y(find(y==0)) = 1;
    x(find(y>0),:) = normrnd(2,1,[size(find(y>0)),2]);
    x(find(y<0),:) = normrnd(-2,1,[size(find(y<0)),2]);
    
%     scatter(x(:,1),x(:,2));
%     hold on;
%     scatter(x(find(y<0),1),x(find(y<0),2),'d');
end







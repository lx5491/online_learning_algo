%% Author: Xi Liu
function [precision] = silk(x, y, kernel_sigma, silk_tau, C)
    addpath('./helper/');
    addpath('./data/handwritten_for_classification');
%     load('data/mnist_49_3000.mat');

    % Change the filenames if you've saved the files under different names
    % On some platforms, the files might be saved as 
    % train-images.idx3-ubyte / train-labels.idx1-ubyte
    % images = loadMNISTImages('data/train-images-idx3-ubyte');
    % labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
    % d = size(images, 1);
    % imagesc(reshape(images(:,3), [sqrt(d), sqrt(d)])');
    % disp(labels(3));

    % x = images;
    % y = labels';
%     [x, y] = get_handwritten(1, 2);

%     [x, y] = gen_data();
    
%     x = x';
%     y = y';
%     size_choose = 10000;
%     x = x(1:size_choose, :);
%     y = y(1:size_choose, :);

%     silk_tau = 0.00005;
%     C = 100;
%     kernel_sigma = 2;
    n = size(x, 1);
    loss_func = 'square';
    alphas = [];

    t = 1;
    correct = 0;
    while t <= n
        x_t = x(t, :);
        y_t = y(t, :);
        if t == 1
            g_x = 1;
            kernel_t = kernel_gaussian(x_t, x_t, kernel_sigma);
            alphas = [alphas, update_t_alpha(kernel_t, y_t, g_x, silk_tau, C, loss_func)];
        else
            k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
            g_x = alphas * k_mat;
            kernel_t = kernel_gaussian(x_t, x_t, kernel_sigma);
            alpha_t = update_t_alpha(kernel_t, y_t, g_x, silk_tau, C, loss_func);
            alphas = [alphas, alpha_t];
        end
        if g_x * y_t > 0
            correct = correct + 1;
        end
%         disp([t, correct]);
        precision = correct / t;
        t = t + 1;
    end
end

function [ alpha ] = update_t_alpha( kernel_t, y_t, g_x, tau, C, loss_func )
    
    if strcmp(loss_func, 'square')
        nom = C * (1 - tau) * (y_t - (1 - tau) * g_x );
        denom = 1 + C * (1 - tau) * kernel_t;
        alpha = nom / denom;
    elseif strcmp(loss_func, 'binary_hinge')
        rho = 1;
        nom = y_t * (rho - (1 - tau) * y_t * g_x);
        denom = kernel_t;
        alpha_tilde = nom / denom;
        if y_t * alpha_tilde < 0
            alpha = 0;
        elseif y_t * alpha_tilde > (1 - tau) * C
            alpha = y_t * (1 - tau) * C;
        else
            alpha = alpha_tilde;
        end
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




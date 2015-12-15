load('mnist_49_3000.mat');
% load('data/BankNote.mat');
% x = Note(:, 1:4);
% y = Note(:, 5);
x = x';
y = y';
size_choose = 3000;
x = x(1:size_choose, :);
y = y(1:size_choose, :);

% code for loading BankNote.mat
% idx = randperm(1372);
% x = x(idx, :);
% y = y(idx, :);

do_truncation = 0; % indicate whether to do truncation
tau = 100; % the number of data to "remember"

eta = 0.0001;
rho = 0;
lambda = 1;
nu = 0.01;
kernel_sigma = 16;
b = 0;
n = size(x, 1);

alphas = [];

t = 1;
correct = 0;
while t <= n
    eta = 0.0001 / sqrt(t);
    if t == 1
        g_x = 1;
        alphas = [alphas, -eta * loss_gradient_sm(g_x, y(t, :), rho)];
    else
        % If do_truncation, need to truncate k_mat to 
        if do_truncation
            x_taus = x(max(1, t - tau): t - 1);
            k_mat = kernel_gaussian(x_taus, x(t, :), kernel_sigma);
        else
            k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
        end
        g_x = alphas * k_mat;
        alphas = (1 - eta * lambda) * alphas;
        if do_truncation && t - tau >= 1
            alphas = [alphas(:, 2:end), -eta * loss_gradient_sm(g_x, y(t, :), rho)];
        else
            alphas = [alphas, -eta * loss_gradient_sm(g_x, y(t, :), rho)];
        end
    end
    if g_x * y(t, :) > 0
        correct = correct + 1;
    end
    disp([t, correct]);
    t = t + 1;
end











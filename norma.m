load('mnist_49_3000.mat');
x = x';
y = y';

eta = 0.1;
rho = 1;
lambda = 1;
nu = 0.01;
kernel_sigma = 16;

n = size(x, 1);
n_train = 2000;
n_test = n - n_train;
x_train = x(1:n_train, :);
x_test = x(n_train + 1: n, :);
y_train = y(1:n_train, :);
y_test = y(n_train + 1: n, :);

alphas = [];

x_tmp = x(2001, :);
a = kernel_gaussian(x_train, x_tmp, kernel_sigma);

t = 1;
while t <= n
    disp(t);
    if t == 1
        f_x = 1;
        alphas = [alphas, -eta * loss_gradient_sm(f_x, y(t, :), rho)];
    else
        k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
        f_x = alphas * k_mat;
        alphas = (1 - eta * lambda) * alphas;
        alphas = [alphas, -eta * loss_gradient_sm(f_x, y(t, :), rho)];
    end
    
    t = t + 1;
end













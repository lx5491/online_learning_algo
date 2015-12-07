load('mnist_49_3000.mat');
x = x';;
y = y';
x = x(1:1500, :);
y = y(1:1500, :);

eta = 0.0001;
rho = 0;
lambda = 1;
nu = 0.01;
kernel_sigma = 16;
n = size(x, 1);

alphas = [];

t = 1;
correct = 0;
while t <= n
    if t == 1
        f_x = 1;
        alphas = [alphas, -eta * loss_gradient_sm(f_x, y(t, :), rho)];
    else
        k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
        f_x = alphas * k_mat;
        alphas = (1 - eta * lambda) * alphas;
        alphas = [alphas, -eta * loss_gradient_sm(f_x, y(t, :), rho)];
    end
    if f_x * y(t, :) > 0
        correct = correct + 1;
    end
    disp([t, correct]);
    t = t + 1;
end













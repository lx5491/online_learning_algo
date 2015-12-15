load('mnist_49_3000.mat');

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
% images = loadMNISTImages('data/train-images-idx3-ubyte');
% labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
% d = size(images, 1);
% imagesc(reshape(images(:,3), [sqrt(d), sqrt(d)])');
% disp(labels(3));

label = label';
images = images';

x = x';
y = y';
size_choose = 3000;
x = x(1:size_choose, :);
y = y(1:size_choose, :);

tau = 0.00005;
C = 100;
kernel_sigma = 16;
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
        alphas = [alphas, update_t_alpha(kernel_t, y_t, g_x, tau, C, loss_func)];
    else
        k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
        g_x = alphas * k_mat;
        kernel_t = kernel_gaussian(x_t, x_t, kernel_sigma);
        alpha_t = update_t_alpha(kernel_t, y_t, g_x, tau, C, loss_func);
        alphas = [alphas, alpha_t];
    end
    if g_x * y_t > 0
        correct = correct + 1;
    end
    disp([t, correct]);
    t = t + 1;
end







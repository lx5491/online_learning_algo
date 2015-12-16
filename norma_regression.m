load('bodyfat_data.mat');
% load('data/BankNote.mat');
% x = Note(:, 1:4);
% y = Note(:, 5);
x = X;
y = y;
% size_choose = 3000;
% x = x(1:size_choose, :);
% y = y(1:size_choose, :);

% code for loading BankNote.mat
% idx = randperm(1372);
% x = x(idx, :);
% y = y(idx, :);

do_truncation = 0; % indicate whether to do truncation
tau = 100; % the number of data to "remember"
loss_func= 'insensitive';
eta = 0.0001;
rho = 0;
lambda = 1;
nu = 0.01;
sigma=0;
epsi=0;
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
        delta=y(t, :)-g_x;
        alphas = [alphas, norma_update_t_alpha( delta,eta, epsi, sigma, loss_func )];
        
        los=loss(delta,epsi,sigma,nu,loss_func);
        
    else
        % If do_truncation, need to truncate k_mat to 
        if do_truncation
            x_taus = x(max(1, t - tau): t - 1);
            k_mat = kernel_gaussian(x_taus, x(t, :), kernel_sigma);
        else
            k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
        end
        g_x = alphas * k_mat;
        delta=y(t, :)-g_x;
        alphas = (1 - eta * lambda) * alphas;
        if do_truncation && t - tau >= 1
            alphas = [alphas(:, 2:end),  norma_update_t_alpha(delta,eta, epsi, sigma, loss_func )];
        else
            alphas = [alphas, norma_update_t_alpha( delta,eta, epsi, sigma, loss_func )];
        end
        
        los=loss(delta,epsi,sigma,nu,loss_func);
        
        if strcmp(loss_func, 'insensitive')
            epsi=update_paremeter(eta,delta,epsi,sigma,nu,loss_func);
        elseif strcmp(loss_func, 'hubers_robust')
            sigma=update_paremeter(eta,delta,epsi,sigma,nu,loss_func);
        end
%       disp([t, g_x, delta])  
    end
    
  disp([t, los]);
    t = t + 1;
end
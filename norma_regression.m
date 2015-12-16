function [] = norma_regression()
    addpath('./helper/');
    load('data/bodyfat_data.mat');
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
end

function [ alpha ] = norma_update_t_alpha(delta,eta, epsi, sigma, loss_func )
    if delta>=0
         sgn=1;
     else
         sgn=-1;
    end

     if strcmp(loss_func, 'square')
        alpha = eta*delta;
    elseif strcmp(loss_func, 'insensitive')
        if abs(delta)>epsi
            alpha=eta*sgn;
        else
            alpha=0;
        end
     elseif strcmp(loss_func, 'hubers_robust')
         if abs(delta)>sigma
            alpha=eta*sgn;
        else
            alpha=delta/sigma;
         end
    end
end

function [result]=update_paremeter(eta,delta,epsi,sigma,nu,loss_func)

    if strcmp(loss_func, 'insensitive')
      if abs(delta)>epsi
          result=epsi+(1-nu)*eta; 
      else
          result=epsi-eta*nu;
      end

    elseif strcmp(loss_func, 'hubers_robust')

        if abs(delta)>sigma
            result=sigma+eta*(1-nu);
        else
            result=sigma-eta*nu;
        end
    end    
end

function [loss]=loss(delta,epsi,sigma,nu,loss_func)

   if strcmp(loss_func, 'square')
         loss= 0.5*delta^2;
    elseif strcmp(loss_func, 'insensitive')
        loss=max(0,abs(delta)-epsi)+nu*epsi;
     elseif strcmp(loss_func, 'hubers_robust')
         if abs(delta)>=sigma
             loss=abs(delta)-0.5*sigma;
         else
             loss=delta^2/(2*sigma);
         end
    
    end
end
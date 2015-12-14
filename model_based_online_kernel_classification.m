function main()
C = 0.8;
r = 0.001;
rho = 1.2;
model_based_classification(C,r,rho);
end

function model_based_classification(C,r,rho)%model based classification
    %input vectors%
    load('mnist_49_3000.mat');
    x = x';
    y = y';
    x = x(1:1500, :);
    y = y(1:1500, :);
    n = size(x,1);
    kernel_sigma = 16;
    %define tunning parameter%
    %C = 10%0.8;
    %r = 0.001;
    %rho = 1.2;
    
    alphas = [];
    t = 1;
    correct = 0;
    while t <= n
        if t == 1
            alpha_new = get_new_alpha(C,r,y(t,:),0);
            f = alpha_new;
            alphas = [alphas;alpha_new];
        else
            %get f_t(x_{t})
            k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
            f = alphas'*k_mat;
            %get alpha_{tt}
            alpha_new = get_new_alpha(C, r, y(t,:),f);
            %update the old alphas first
            alphas = alphas.*(1/(1+r));
            alphas = [alphas;alpha_new];
        end
        if f*y(t,:) > 0
            correct = correct + 1;
        end
        t = t + 1;
        fprintf('%d, %d\n',t,correct);
    end
    fprintf('C = %f,r = %f,rho = %f, correct = %f\n',C,r,rho,correct/n);

end

function new_alpha = get_new_alpha(C,r,y,f)
    alpha_bar = 1 + r - y*f;
    if alpha_bar < 0
        new_alpha = 0;
    else
        if alpha_bar >= C
            new_alpha = C/(1+r)*y;
        else
            new_alpha = (1+r-y*f)*y/(1+r);
        end
    end
end





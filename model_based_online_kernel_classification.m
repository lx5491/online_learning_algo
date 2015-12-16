
 
function model_based_online_kernel_classification(C,r,rho,x,y,kernel_sigma)%model based classification
    %tunning parameter set to be :
    %C = 1; 
    %r = 0.001;
    %rho = 1.2;
    %kernel_sigma = 1600;
    %change one of the class to -1
    indices = randperm(size(x,1));
    x = x(indices, :);
    y = y(indices, :);
    n = size(x,1);
   
    alphas = [];
    t = 1;
    correct = 0;
    while t <= n
        if t == 1
            
            f = 0 ;
            alpha_new = get_new_alpha(C,r,y(t,:),f);
        
            alphas = [alphas;alpha_new];
        else
            %get f_t(x_{t+1})
            k_mat = kernel_gaussian(x(1:t-1, :), x(t, :), kernel_sigma);
            f = alphas'*k_mat;%eqn 31
            
            %get alpha_{t+1}
            alpha_new = get_new_alpha(C, r, y(t,:),f);
            
            %update the old alphas first
            alphas = alphas/(1+r);%eqn 33
            %put alpha_{t+1} into alphas
            alphas = [alphas;alpha_new];
        end
        %prediction error
        if f*y(t,:) > 0
            correct = correct + 1;
        end
        %fprintf('f = %f, y = %d, t = %d, correct= %d\n',f, y(t,:),t,correct);
       
        t = t + 1;
    end
    fprintf('C = %f,r = %f,rho = %f, errors = %d out of %d\n',C,r,rho,t - correct -1, t -1);

end

function new_alpha = get_new_alpha(C,r,y,f)
    alpha_bar = 1 + r - y*f;%sentence after eqn 30
    %eqn 33
    if alpha_bar <= 0
        new_alpha = 0;
    else
        if alpha_bar >= C
            new_alpha = C/(1+r)*y;
        else
            new_alpha = (1+r-y*f)*y/(1+r);
        end
    end
end





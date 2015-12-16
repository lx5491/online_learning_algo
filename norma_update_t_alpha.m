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
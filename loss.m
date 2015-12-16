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
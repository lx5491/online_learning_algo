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
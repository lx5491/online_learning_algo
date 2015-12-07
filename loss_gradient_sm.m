function [ result ] = loss_gradient_sm( f_x, y, rho )
    if f_x * y <= rho
        result = -y;
    else
        result = 0;
    end
end


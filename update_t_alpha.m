function [ alpha ] = update_t_alpha( kernel_t, y_t, g_x, tau, C, loss_func )
    if strcmp(loss_func, 'square')
        nom = C * (1 - tau) * (y_t - (1 - tau) * g_x );
        denom = 1 + C * (1 - tau) * kernel_t;
        alpha = nom / denom;
    elseif strcmp(loss_func, 'binary_hinge')
        rho = 0;
        nom = y_t * (rho - (1 - tau) * y_t * g_x);
        denom = kernel_t;
        alpha_tilde = nom / denom;
        if y_t * alpha_tilde < 0
            alpha = 0;
        elseif y_t * alpha_tilde > (1 - tau) * C
            alpha = y_t * (1 - tau) * C;
        else
            alpha = alpha_tilde;
        end
    end
end

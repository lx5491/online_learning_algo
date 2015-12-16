function [ kernel_mat ] = kernel_gaussian( X1, X2, sigma )
    norm_square_vec = dist2(X1, X2);
    kernel_mat = exp(- norm_square_vec ./ (2 * sigma^2));
end

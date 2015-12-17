% Author: Xi Liu

function [] = experiment()
    addpath('./data/handwritten_for_classification');
    %run_norma_experiment();
    %run_silk_experiment();
    run_model_based_online_experiment();
end

function [] = run_norma_experiment()
    norma_file_id = fopen('norma_results.txt','w');
    fprintf(norma_file_id, 'data_set, kernel_sigma, rho, nu, do_truncation, tau, norma_precision, runtime\n');

    rhos = [0, 0.1];
    nus = [0.01, 0.05, 0.1];
    % kernel_sigmas = [4, 16, 64, 256, 1024, 1500, 2048];
    kernel_sigmas = [2000, 1500, 1000, 200, 100, 16, 4];
    % kernel_sigmas = [1600];
    do_truncations = [0, 1];
    % do_truncations = [0];
    taus = [100, 500, 1000];

    [x_mnist_12, y_mnist_12] = get_handwritten(1, 2);
    [x_mnist_49, y_mnist_49] = get_handwritten(4, 9);
    [x_mnist_56, y_mnist_56] = get_handwritten(5, 6);
    disp('Data loading finished');

    for i = 1:3
        if i == 1
            x = x_mnist_12;
            y = y_mnist_12;
            data_set = 'mnist_12';
        elseif i == 2
            x = x_mnist_49;
            y = y_mnist_49;
            data_set = 'mnist_49';
        elseif i == 3
            x = x_mnist_56;
            y = y_mnist_56;
            data_set = 'mnist_56';
        end
        for kernel_sigma = kernel_sigmas
            for rho = rhos
                for nu = nus
                    for do_truncation = do_truncations
                        if do_truncation == 1
                            for tau = taus
                                fprintf('Running %s, %d, %f, %f, %d, %d\n', data_set, kernel_sigma, rho, nu, do_truncation, tau);
                                tic;
                                [norma_precision] = norma(x, y, kernel_sigma, rho, nu, do_truncation, tau);
                                elapse_runtime = toc;
                                fprintf(norma_file_id, '%s, %d, %f, %f, %d, %d, %f, %f\n', data_set, kernel_sigma, rho, nu, do_truncation, tau, norma_precision, elapse_runtime);
                            end
                        else
                            tau = 0;
                            fprintf('Running %s, %d, %f, %f, %d, %d\n', data_set, kernel_sigma, rho, nu, do_truncation, tau);
                            tic;
                            [norma_precision] = norma(x, y, kernel_sigma, rho, nu, do_truncation, tau);
                            elapse_runtime = toc;
                            fprintf(norma_file_id, '%s, %d, %f, %f, %d, %d, %f, %f\n', data_set, kernel_sigma, rho, nu, do_truncation, tau, norma_precision, elapse_runtime);
                        end
                    end
                end
            end
        end
    end

    fclose(norma_file_id);
end

function [] = run_silk_experiment()
    silk_file_id = fopen('silk_results.txt','w');
    fprintf(silk_file_id, 'data_set, kernel_sigma, silk_tau, C, silk_precision, runtime\n');

    % kernel_sigmas = [4, 16, 64, 256, 1024, 1500, 2048];
    kernel_sigmas = [2000, 1500, 1000, 200, 100, 16, 4];
    silk_taus = [0.00001, 0.00005, 0.0001, 0.0005];
    Cs = [10, 50, 100, 200];

    [x_mnist_12, y_mnist_12] = get_handwritten(1, 2);
    [x_mnist_49, y_mnist_49] = get_handwritten(4, 9);
    [x_mnist_56, y_mnist_56] = get_handwritten(5, 6);
    disp('Silk Data loading finished');

    for i = 1:3
        if i == 1
            x = x_mnist_12;
            y = y_mnist_12;
            data_set = 'mnist_12';
        elseif i == 2
            x = x_mnist_49;
            y = y_mnist_49;
            data_set = 'mnist_49';
        elseif i == 3
            x = x_mnist_56;
            y = y_mnist_56;
            data_set = 'mnist_56';
        end
        for kernel_sigma = kernel_sigmas
            for silk_tau = silk_taus
                for C = Cs
                    fprintf('Running %s, %d, %f, %d\n', data_set, kernel_sigma, silk_tau, C);
                    tic;
                    [silk_precision] = silk(x, y, kernel_sigma, silk_tau, C);
                    elapse_runtime = toc;
                    fprintf(silk_file_id, '%s, %d, %f, %d, %f, %f\n', data_set, kernel_sigma, silk_tau, C, silk_precision, elapse_runtime);
                end
            end
        end
    end

    fclose(silk_file_id);
end

function [] = run_model_based_online_experiment()
    silk_file_id = fopen('model_based_online_results.txt','w');
    fprintf(silk_file_id, 'data_set, kernel_sigma, r, C, rho, model_based_precision, runtime\n');

    % kernel_sigmas = [4, 16, 64, 256, 1024, 1500, 2048];
    kernel_sigmas = [2000, 1500, 1000, 200, 100, 16, 4];
    rs = [0.0005, 0.001, 0.005, 0.01];
    Cs = [1, 5, 10, 50, 100];
    rhos = [1.2, 1, 0.5, 0];

    [x_mnist_12, y_mnist_12] = get_handwritten(1, 2);
    [x_mnist_49, y_mnist_49] = get_handwritten(4, 9);
    [x_mnist_56, y_mnist_56] = get_handwritten(5, 6);
    disp('Model Data loading finished');

    for i = 1:3
        if i == 1
            x = x_mnist_12;
            y = y_mnist_12;
            data_set = 'mnist_12';
        elseif i == 2
            x = x_mnist_49;
            y = y_mnist_49;
            data_set = 'mnist_49';
        elseif i == 3
            x = x_mnist_56;
            y = y_mnist_56;
            data_set = 'mnist_56';
        end
        for kernel_sigma = kernel_sigmas
            for r = rs
                for C = Cs
                    for rho = rhos
                        fprintf('Running %s, %d, %f, %d, %f\n', data_set, kernel_sigma, r, C, rho);
                        tic;
                        [model_based_precision] = model_based_online_kernel_classification(C, r, rho, x, y, kernel_sigma);
                        elapse_runtime = toc;
                        fprintf(silk_file_id, '%s, %d, %f, %d, %f, %f, %f\n', data_set, kernel_sigma, r, C, rho, model_based_precision, elapse_runtime);
                    end
                end
            end
        end
    end

    fclose(silk_file_id);
end
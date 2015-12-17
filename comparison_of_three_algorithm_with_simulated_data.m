function [] = experiment()
   
   
   total_ranges = [50:100:3000];
   model_based_precision = zeros(size(total_ranges));
   norma_precision_with_trunc = zeros(size(total_ranges));
   norma_precision_wo_trunc = zeros(size(total_ranges));

   silk_precision = zeros(size(total_ranges));
   model_based_runtime = zeros(size(total_ranges));
   norma_runtime_with_trunc = zeros(size(total_ranges));
   norma_runtime_wo_trunc = zeros(size(total_ranges));

   silk_runtime = zeros(size(total_ranges));
   
   for t = 1 : size(total_ranges,2)
        total_range =total_ranges(t);
        x = zeros(total_range,2);
        y = zeros(total_range, 1);
        umin=-1;
        umax=1;
        u=umin+rand(1,size(x,1))*(umax-umin);
        y = sign(u)';
        y(find(y==0)) = 1;
        x(find(y>0),:) = normrnd(1,1,[size(find(y>0)),2]);
        x(find(y<0),:) = normrnd(-1,1,[size(find(y<0)),2]);
        figure(1);
        if t == size(total_ranges,2)
            scatter(x(find(y>0),1), x(find(y>0),2), 'b');
            hold on;
            scatter(x(find(y<0),1), x(find(y<0),2),'r');
            hold off;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        kernel_sigma = 0.5;
        C = 1; 
        r = 0.001;
        tic; 
        model_based_precision(t) = model_based_online_kernel_classification(C, r, x, y, kernel_sigma);
        elapse_runtime = toc;
        model_based_runtime(t) = elapse_runtime;
        %%%%%%%%%%%%%%%%%%%%%%%%
        do_truncation = 1; % indicate whether to do truncation
        tau = 1000; % the number of data to "remember"
        rho = 0;
        nu = 0.01;
        kernel_sigma = 1;
        tic; 
        norma_precision_with_trunc(t) = norma(x, y, kernel_sigma, rho, nu, do_truncation, tau);
        elapse_runtime = toc;
        norma_runtime_with_trunc(t) = elapse_runtime;
        
        do_truncation = 0;
        tic; 
        norma_precision_wo_trunc(t) = norma(x, y, kernel_sigma, rho, nu, do_truncation, tau);
        elapse_runtime = toc;
        norma_runtime_wo_trunc(t) = elapse_runtime;
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        kernel_sigma = 1;
        silk_tau = 0.00001; 
        C = 0.5;
        tic; 
        silk_precision(t) = silk(x, y, kernel_sigma, silk_tau, C);
        elapse_runtime = toc;
        silk_runtime(t) = elapse_runtime;
        
   end
   figure(2);
   plot(total_ranges, norma_precision_with_trunc,'b--o',total_ranges, norma_precision_wo_trunc,'r--o',total_ranges,model_based_precision,'c--*',total_ranges,silk_precision,'g--o');
   legend('NORMA with truncation','NORMA without truncation','OLK','ILK');
   xlabel('Number of Data');
   ylabel('Precision');
   figure(3);
   plot(total_ranges, norma_runtime_with_trunc,'b--o',total_ranges, norma_runtime_wo_trunc, 'r--o', total_ranges,model_based_runtime,'c--*',total_ranges,silk_runtime,'g--o');
   legend('NORMA with truncation','NORMA without truncation','OLK','ILK');
   xlabel('Number of Data');
   ylabel('Runtime');
   
   %model_based_precision 
   %norma_precision 
   %silk_precision 
   
        
   
end
   
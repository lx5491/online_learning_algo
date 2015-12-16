function main()
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%test of handwritten datasets
[x,y] = get_handwritten(4,9);
%parameter of norma
do_truncation = 1; % indicate whether to do truncation
tau = 1000; % the number of data to "remember"
eta = 0.0001;
rho = 0;
kernel_sigma = 1000;
lambda = 1;
nu = 0.01;

display('norma.....');
norma(x,y,kernel_sigma, tau, eta, rho, lambda, nu,do_truncation);

%parameter od model-based
display('model_based....')
C = 1; 
r = 0.001;
rho = 1.2;
model_based_online_kernel_classification(C,r,rho,x,y,kernel_sigma);


%%%%%%%%%%%%%
%%%%%%%%%%%%%
%two distribution tests

%generate datasets
total_range =10000;
 x = zeros(total_range,2);
    y = zeros(total_range, 1);
    umin=-1;
    umax=1;
    n=total_range;
    u=umin+rand(1,n)*(umax-umin);
    y = sign(u)';
    y(find(y==0)) = 1;
    x(find(y>0),:) = normrnd(2,1,[size(find(y>0)),2]);
    x(find(y<0),:) = normrnd(-2,1,[size(find(y<0)),2]);
     
    scatter(x(:,1),x(:,2));
    hold on;
    scatter(x(find(y<0),1),x(find(y<0),2),'d');

kernel_sigma = 4;
%parameter of norma
do_truncation = 1; % indicate whether to do truncation
tau = 1000; % the number of data to "remember"
eta = 0.0001;
rho = 0;
lambda = 1;
nu = 0.01;

display('norma.....');
norma(x,y,kernel_sigma, tau, eta, rho, lambda, nu,do_truncation);
%parameter of model-based
display('model_based....')
C = 1; 
r = 0.001;
rho = 1.2;
model_based_online_kernel_classification(C,r,rho,x,y,kernel_sigma);

end
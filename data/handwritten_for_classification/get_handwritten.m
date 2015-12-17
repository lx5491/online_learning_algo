%dataset: http://cis.jhu.edu/~sachin/digit/digit.html
%data_{x} is handwritten pic for x; 
%return X with 2 classes and y with 1 and -1.
%e.g. gethandwritten(1,2) return x of digit 1 and digit 2 feature vectors.

function [x, y] = get_handwritten(num1, num2)
   rng(0);
   data1 = strcat('data',num2str(num1));
   data2 = strcat('data',num2str(num2));
   fid = fopen(data1,'r');
   x = zeros(2000,28*28);
   y = zeros(2000,1);
   for t = 1: 1000
        x(t,:) = reshape(fread(fid,[28 28],'uchar'), [1,28*28]);
   end
   y(1:1000,:) = ones(1000,1);
   
   fid2 = fopen(data2,'r');
   for t = 1001: 2000
        x(t,:) = reshape(fread(fid2,[28 28],'uchar'), [1,28*28]);
   end
   y(1001:2000, :) = -1*ones(1000,1);
   
   %randomize the dataset
   indices = randperm(2000);
   x = x(indices, :);
   y = y(indices, :);
   
   %size(x);
   %imagesc(reshape(x(155,:), [28,28]));
   %y(155,:)
end


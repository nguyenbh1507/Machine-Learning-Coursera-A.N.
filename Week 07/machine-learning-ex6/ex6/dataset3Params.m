function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sizeC = size(C_test,2);
sizeSig = size(sigma_test,2);
result = zeros(sizeC*sizeSig,3); %first column for C, second for sig, third for error
C1 = C_test' * ones(1, sizeSig);
sig2 = sigma_test' * ones(1, sizeC);
result = [C1'(:),sig2(:),zeros(sizeC*sizeSig,1)];
x1 = [1 2 1]; x2 = [0 4 -1]; 

for i = 1:sizeC*sizeSig
  tC = result(i,1);
  tsigma = result(i,2);
  model= svmTrain(X, y, tC, @(x1, x2) gaussianKernel(x1, x2, tsigma));
  pred = svmPredict(model, Xval);
  pred_error = mean(double(pred ~= yval));
  result(i,3) = pred_error;  
endfor;

[val,idx] = min(result(:,3)); %index of minimum error 

C = result(idx,1);
sigma = result(idx,2);


% =========================================================================

end

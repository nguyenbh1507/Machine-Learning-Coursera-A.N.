function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%A1 = [ones(m,1) X]; % A1.size = 5000 x 401
%Z2 = Theta1 * A1';  % theta1.size = 25 x 401; A1'.size = 401 x 5000; 
%A2 = sigmoid(Z2');   % A2.size = 25 x 5000
%%A2 = A2';           % A2.size = 5000 x 25
%A2 = [ones(size(A2, 1),1) A2]; %A2.size = 5000x26
%Z3 = Theta2 * A2';  % theta2.size = 10x26, A2'.size = 26x5000
%A3 = sigmoid(Z3');   % A3.size = 10*5000;
%%A3 = A3';
%[val p] = max(A3,[],2);
%%A1 = [ones(m,1) X]; % A1.size = 5000 x 401

A1 = [ones(m,1) X]; % A1.size = 5000 x 401
Z2 = A1*Theta1';  % theta1.size = 25 x 401; 
A2 = sigmoid(Z2);   % A2.size = 5000 x 25

A2 = [ones(size(A2, 1),1) A2]; %A2.size = 5000x26
Z3 = A2 * Theta2';  % theta2.size = 10x26
A3 = sigmoid(Z3);   % A3.size = 5000x10;

[val p] = max(A3,[],2);











% =========================================================================


end

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



C_list=[0.01 0.03 0.1 0.3 1, 3, 10 30];

s_list=[0.01 0.03 0.1 0.3 1, 3, 10 30];

results = zeros(length(C_list) * length(s_list), 3);
row = 1;
for i=1:length(C_list)
    for j=1:length(s_list)
      model=svmTrain(X,y,C_list(i),@(x1,x2)gaussianKernel(x1,x2,s_list(j)));
      
      predictions = svmPredict(model, Xval);
%The implementation of cost function for SVM in Octave is as simple as below:
%mean(double(predictions ~= yval))
%Let me explain it a little bit:
%“mean” function is to calculate the mean of the cross validation dataset. 
%“double” function is to convert the datatype to double. 
%“~=” is an operator to implement a logic that for each cross validation data, 
%if the prediction is same with yval, return 0, otherwise return 1;
      prediction_error = mean(double(predictions ~= yval));
     % size(predictions)
    % size(yval)
 %prediction_error(:)
       results(row,:)=[C_list(i),s_list(j),prediction_error];
      row=row+1;
    endfor
endfor
sorted_results = sortrows(results, 3); % sort matrix by column #3, the error, ascending

C = sorted_results(1,1);
sigma = sorted_results(1,2);


% =========================================================================

end

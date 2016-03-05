function [ val,grad ] = G3Function(u,lambda)
%G3Function sum total of the evaluation of the function(g3) and its 
% gradient at  each datapoint u

uAbs = abs(u);

val = sum(sum(lambda*uAbs - (lamda^2)*log(1+uAbs/lambda)));
grad = 0.5*lambda/(lambda+uAbs);

end


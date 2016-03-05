function [x,logCostArray,iter] = GradientDescent(xInit,y,g,maxIters,lambda)
%GradientDescent Implements adaptive gradient descent using potential function
%for MRF and other params passed as arguments

x = xInit;
stepSize=0.1; % Initial step size
gradientThreshold = 1e-4; % Threshold for gradient change

logCostArray = zeros(maxIters,1); 

[logCost2,grad2] = MRFEval(x,g);
[logCost1,grad1] = GetLikelihoodTerm(x,y);

logCost = lambda*logCost1+(1-lambda)*logCost2;
grad = lambda*grad1+(1-lambda)*grad2;
logCostArray(1)=logCost;

iter=1;
while(1)

    if max(max(abs(grad)))<gradientThreshold || iter>=maxIters || stepSize<1e-5
        break
    end
    xNew = x-stepSize.*grad;
    
    [newLogCost2,newGrad2] = MRFEval(xNew,g);
    [newLogCost1,newGrad1] = GetLikelihoodTerm(xNew,y);
    
    newLogCost = lambda*newLogCost1+(1-lambda)*newLogCost2;
    newGrad = lambda*newGrad1+(1-lambda)*newGrad2;
    
    if newLogCost<logCost
        x = xNew;
        stepSize = stepSize*1.1;
        iter=iter+1;
        logCost = newLogCost;
        grad = newGrad;
        logCostArray(iter)=logCost;
    else
        stepSize = stepSize*0.5;
    end  
end


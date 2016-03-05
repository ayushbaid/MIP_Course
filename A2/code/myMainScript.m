% Denoising a phantom MRI image

%% Loading the data
load('../data/assignmentImageDenoisingPhantom.mat');

%% A) RRSME of given noisy image

noiselessNorm = sqrt(sumsqr(abs(imageNoiseless)));
initialRRMSE = sqrt(sumsqr(abs(imageNoiseless)-abs(imageNoisy)))/noiselessNorm;

%% B) 1: Using quadratic function
g = @(x) QuadraticFunction(x);
lambdaRange = 0:0.05:1;
rrmse = zeros(length(lambdaRange),1);

for i=1:length(lambdaRange)
    lambda = lambdaRange(i);

    [x,logCostArray,iters] = GradientDescent(imageNoisy,imageNoisy,g,100,lambda);
    rrmse(i) = sqrt(sumsqr(abs(imageNoiseless)-abs(x)))/noiselessNorm;
%     figure(1);
%     plot(logCostArray(1:iters));
%     title('Log cost function');
end




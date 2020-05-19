function data_feature = extract_features(X, centroids,ActivationFunction)
[row, col, num_PC] = size(X);
K = size(centroids,2);
w = sqrt(size(centroids,1)/num_PC);
row = row - w+1;
col = col - w+1;

data_feature = zeros(row*col,K);
filter = reshape(centroids,w,w,num_PC,K);
XC = zeros(row*col,K);

for i=1:K
    for j=1:num_PC
        filter0 = filter(:,:,j,i);
        filter0 = rot90(squeeze(filter0),2);
        
        img = squeeze(X(:, :, j));
        
        convolvedImage = conv2(img,filter0,'valid');
        
        XC(:,i) = XC(:,i) + convolvedImage(:);
    end
end
%CC = sum(centroids.^2);
Z= XC;
% Z = -bsxfun(@minus, XC,0.5*CC);%z :m*K
mu = mean(Z, 2);

if strcmp(ActivationFunction,'Relu')
    data_feature = max(bsxfun(@minus, Z, mu), 0);%Relu
elseif strcmp(ActivationFunction,'LeakyRelu')
    data_feature = max(bsxfun(@minus, Z, mu), 0.01.*bsxfun(@minus, Z, mu));%LeakyReLU
else 
    error('Error ActivationFunction');
end

% data_feature = max(bsxfun(@minus, Z, mu), 0);%Relu
% data_feature = max(bsxfun(@minus, Z, mu), 0.01.*bsxfun(@minus, Z, mu));%LeakyReLU
% data_feature = max(bsxfun(@minus, Z, mu), 1.6.*(exp(bsxfun(@minus, Z, mu))-1));%ELU

% CC = sum(centroids.^2);
% Z = bsxfun(@plus, -2*XC,CC);%z :m*K
% mu = mean(Z, 2);
% data_feature = max(bsxfun(@minus, mu, Z), 0);

end
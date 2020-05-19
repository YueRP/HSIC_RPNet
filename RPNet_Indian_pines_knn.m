%% Load data
clear;
close all;
tic
data_name='Indian_pines_corrected.mat';
data_gt='Indian_pines_gt.mat';
% data_name='PaviaU.mat';
% data_gt='PaviaU_gt.mat';
addpath(genpath('utils'))
addpath(genpath('dataset'))
a=load(data_name);
% a = load('Indian_pines_corrected.mat');
a=load(data_name);
Data = a.data;
[row,col,num_feature] = size(Data);
a=load(data_gt);
% a = load('Indian_pines_gt.mat');
Label = reshape(double(a.groundT),row*col,1);
num_class = max(Label(:));
clear a;

train_num_array = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50];
train_num_all = sum(train_num_array);

% no_dims = round(intrinsic_dim(reshape(Data, row * col, num_feature),
% 'MLE'));% 估计本质维数 此处为10

num_PC = 3;

Layernum = 5;

w=21;
win_inter = (w-1)/2;
epsilon = 0.01;
K=20;

StackFeature= cell(Layernum,1);

dr='MDS';
KNN=1
repeat=1;
OA_sum=zeros(1,repeat);
Kappa_sum=zeros(1,repeat);

for ii=1:repeat
disp(['the',num2str(ii)]);    

for l=1:Layernum
    
    randidx = randperm(row*col);
    StackFeature{l}.centroids = zeros(w*w*num_PC,K);
%     disp(['Extracting the features of the ',num2str(l),'th layer...']);
    if l==1
        
        XPCA = compute_mapping(reshape(Data, row * col, num_feature),dr,num_PC);
        
        XPCAvector = XPCA;
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        
        XPCA_cov = cov(XPCA);
        [U S V] = svd(XPCA_cov);
        whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
        XPCA = XPCA * whiten_matrix;
        XPCA = bsxfun(@rdivide,bsxfun(@minus,XPCA,mean(XPCA,1)),std(XPCA,0,1)+epsilon);
        XPCA = reshape(XPCA,row,col,num_PC);
        X_extension = MirrowCut(XPCA,win_inter);
        
        for i=1:K
            index_col = ceil(randidx(i)/row);
            index_row = randidx(i) - (index_col-1) * row;
            tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
            StackFeature{l}.centroids(:,i) = tem(:);
        end
        
        StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids);
        
        XPCAvector = compute_mapping([StackFeature{l}.feature],dr,num_PC);
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        clear StackFeature{l}.centroids;
    else
        XPCA = compute_mapping(StackFeature{l-1}.feature,dr,num_PC);
        
        XPCA_cov = cov(XPCA);
        [U S V] = svd(XPCA_cov);
        whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
        
        
        XPCA = XPCA * whiten_matrix;
        XPCA = bsxfun(@rdivide,bsxfun(@minus,XPCA,mean(XPCA,1)),std(XPCA,0,1)+epsilon);
        
        XPCA = reshape(XPCA,row,col,num_PC);
        X_extension = MirrowCut(XPCA,win_inter);
        
        for i=1:K
            index_col = ceil(randidx(i)/row);
            index_row = randidx(i) - (index_col-1) * row;
            tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
            StackFeature{l}.centroids(:,i) = tem(:);
        end
        
        StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids);
        
        XPCAvector = compute_mapping(StackFeature{l}.feature,dr,num_PC);
        minZ = min(XPCAvector);
        maxZ = max(XPCAvector);
        XPCAvector = bsxfun(@minus, XPCAvector, minZ);
        XPCAvector = bsxfun(@rdivide, XPCAvector, maxZ-minZ);
        
        clear StackFeature{l}.centroids;
    end
    
    clear X_extension;
end

%%
% for layernum=1:Layernum
for layernum=Layernum
    
    X_joint = [];
    for i=1:layernum
        X_joint = [X_joint StackFeature{i}.feature];
    end
    X_joint = [X_joint reshape(Data,row*col,num_feature)];
    X_joint_mean = mean(X_joint);
    X_joint_std = std(X_joint)+1;
    X_joint = bsxfun(@rdivide, bsxfun(@minus, X_joint, X_joint_mean), X_joint_std);
    
    randomLabel = cell(num_class,1);
    for i=1:num_class
        index = find(Label==i);
        randomLabel{i}.array = randperm(size(index,1));
    end
        
    X_train = [];
    X_test = [];
    y_train = [];
    y_test = [];
    
    for i=1:num_class
        index = find(Label==i);
        randomX = randomLabel{i,1}.array;
        train_num = train_num_array(i);
        X_train = [X_train;X_joint(index(randomX(1:train_num)),:)];
        y_train = [y_train;Label(index(randomX(1:train_num)),1)];
        
        X_test = [X_test;X_joint(index(randomX(train_num+1:end)),:)];
        y_test = [y_test;Label(index(randomX(train_num+1:end)),1)];
        
    end
   
    model=fitcknn(X_train,y_train,'NumNeighbors',KNN);
    label = predict(model,X_test);
    [OA Kappa producerA] = CalAccuracy(label,y_test);
    OA_sum(ii)=OA;
    Kappa_sum(ii)=Kappa;
    
    labels=predict(model,X_joint);
     X_result = drawresult(labels,row,col, 2);
    imwrite(X_result,strcat('RPNet_knn_Indian_',num2str(KNN),'.png'),'png');
end
end
OA_average=mean(OA_sum);
Kappa_average=mean(Kappa_sum);
toc
disp(['OA= ',num2str(OA_average),' kappa=',num2str(Kappa_average),' time=',num2str(toc/repeat)]);

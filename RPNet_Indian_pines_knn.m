clear ;
close all;
tic     %开始计时

repeat=2;   %实验重复次数

%手动设置降维参数以及激活函数
dr='MDS';   %降维方法 MDS,PCA,LDA,FA

ActivationFunction='LeakyRelu';
% ActivationFunction='Relu';

KNN=1;%KNN中的K值
num_PC = 3; %降维参数
Layernum = 5;    %网络层数
w=21;   %随机块大小
win_inter = (w-1)/2;
epsilon = 0.01;
K=20;   %随机块数量


%初始化每次实验的OA与Kappa为0
OA_set=zeros(1,repeat);
Kappa_set=zeros(1,repeat);

% 加载数据
data_name='Indian_pines_corrected.mat';
data_gt='Indian_pines_gt.mat';
addpath(genpath('utils'))
addpath(genpath('dataset'))
a=load(data_name);
Data = a.data;
[row,col,num_feature] = size(Data);
a=load(data_gt);%ground truth
Label = reshape(double(a.groundT),row*col,1);
num_class = max(Label(:));
clear a;

train_num_array = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50];%训练集划分
train_num_all = sum(train_num_array);


%重复试验取均值
for ii=1:repeat
    disp(['第',num2str(ii),'次实验']);%输出重复次数
    
    StackFeature= cell(Layernum,1);
    for l=1:Layernum
        
        randidx = randperm(row*col);
        StackFeature{l}.centroids = zeros(w*w*num_PC,K);
        % disp(['Extracting the features of the ',num2str(l),'th layer...']);
        
        if l==1
            % Step 1: 对原数据X降维
            XDR = compute_mapping(reshape(Data, row * col, num_feature),dr,num_PC);
            XDRvector = XDR;
            minZ = min(XDRvector);
            maxZ = max(XDRvector);
            XDRvector = bsxfun(@minus, XDRvector, minZ);
            XDRvector = bsxfun(@rdivide, XDRvector, maxZ-minZ);
            
            %Step 2: 计算白化数据 XWhiten
            XDR_cov = cov(XDR);
            [U S V] = svd(XDR_cov);
            whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
            XDR = XDR * whiten_matrix;
            XDR = bsxfun(@rdivide,bsxfun(@minus,XDR,mean(XDR,1)),std(XDR,0,1)+epsilon);
            XDR = reshape(XDR,row,col,num_PC);
            X_extension = MirrowCut(XDR,win_inter);
            %从白化数据XWhiten中提取k个随机块.
            for i=1:K
                index_col = ceil(randidx(i)/row);
                index_row = randidx(i) - (index_col-1) * row;
                tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
                StackFeature{l}.centroids(:,i) = tem(:);
            end
            
            StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids,ActivationFunction);
            
            XDRvector = compute_mapping([StackFeature{l}.feature],dr,num_PC);
            minZ = min(XDRvector);
            maxZ = max(XDRvector);
            XDRvector = bsxfun(@minus, XDRvector, minZ);
            XDRvector = bsxfun(@rdivide, XDRvector, maxZ-minZ);
            
            clear StackFeature{l}.centroids;
        else
            XDR = compute_mapping(StackFeature{l-1}.feature,dr,num_PC);
            
            XDR_cov = cov(XDR);
            [U S V] = svd(XDR_cov);
            whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
            
            XDR = XDR * whiten_matrix;
            XDR = bsxfun(@rdivide,bsxfun(@minus,XDR,mean(XDR,1)),std(XDR,0,1)+epsilon);
            
            XDR = reshape(XDR,row,col,num_PC);
            X_extension = MirrowCut(XDR,win_inter);
            
            for i=1:K
                index_col = ceil(randidx(i)/row);
                index_row = randidx(i) - (index_col-1) * row;
                tem = X_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
                StackFeature{l}.centroids(:,i) = tem(:);
            end
            
            StackFeature{l}.feature = extract_features(X_extension,StackFeature{l}.centroids,ActivationFunction);
            
            XDRvector = compute_mapping(StackFeature{l}.feature,dr,num_PC);
            
            minZ = min(XDRvector);
            maxZ = max(XDRvector);
            XDRvector = bsxfun(@minus, XDRvector, minZ);
            XDRvector = bsxfun(@rdivide, XDRvector, maxZ-minZ);
            
            clear StackFeature{l}.centroids;
        end
        
        clear X_extension;
    end
    
    
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
        
        
        %训练KNN
        model=fitcknn(X_train,y_train,'NumNeighbors',KNN);
        %预测
        label = predict(model,X_test);
        %计算OA，Kappa，以及每个类别的准确率
        [OA, Kappa, producerA] = CalAccuracy(label,y_test);
        
        
        %保存OA，Kappa
        OA_set(ii)=OA;
        Kappa_set(ii)=Kappa;
        
        labels=predict(model,X_joint);
        X_result = drawresult(labels,row,col, 2);
        
        %输出预测图像
        str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
        str1=strcat('Indian_Pines_',dr,'_knn_',ActivationFunction);
        str2='.png';
        save_path=[str0,str1,str2];
        imwrite(X_result,save_path);
    end
end

%计算OA，Kappa的均值
OA_average=mean(OA_set);
Kappa_average=mean(Kappa_set);
%输出OA，Kappa，time
disp(['分类器:KNN']);
disp(['降维方法:',dr]);
disp(['降维参数:',num2str(num_PC)]);
disp(['网络层数:',num2str(Layernum)]);
disp(['激活函数:',ActivationFunction]);
disp(['OA= ',num2str(OA_average)]);
disp(['Kappa=',num2str(Kappa_average)]);
disp(['time=',num2str(toc/repeat)]);















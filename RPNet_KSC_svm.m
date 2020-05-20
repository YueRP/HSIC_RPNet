clear ;
close all;
tic     %��ʼ��ʱ

repeat=5;   %ʵ���ظ�����

%�ֶ����ý�ά�����Լ������
dr='MDS';   %��ά���� MDS,PCA,LDA,FA

ActivationFunction='LeakyRelu';
% ActivationFunction='Relu';

num_PC = 10; %��ά����
Layernum = 10;    %�������
w=21;   %������С
win_inter = (w-1)/2;
epsilon = 0.01;
K=20;   %���������


%��ʼ��ÿ��ʵ���OA��KappaΪ0
OA_set=zeros(1,repeat);
Kappa_set=zeros(1,repeat);

% ��������
data_name='KSC.mat';
data_gt='KSC_gt.mat';
addpath(genpath('utils'))
addpath(genpath('dataset'))
a=load(data_name);
Data = a.KSC;
[row,col,num_feature] = size(Data);
a=load(data_gt);%ground truth
Label = reshape(double(a.KSC_gt),row*col,1);
num_class = max(Label(:));
clear a;

train_num_array=[33,23,24,24,15,22,9,38,51,39,41,49,91];%KSCѵ��������
train_num_all = sum(train_num_array);


%�ظ�����ȡ��ֵ
for ii=1:repeat
    disp(['��',num2str(ii),'��ʵ��']);%����ظ�����
    
    StackFeature= cell(Layernum,1);
    for l=1:Layernum
        
        randidx = randperm(row*col);
        StackFeature{l}.centroids = zeros(w*w*num_PC,K);
        % disp(['Extracting the features of the ',num2str(l),'th layer...']);
        
        if l==1
            % Step 1: ��ԭ����X��ά
            XDR = compute_mapping(reshape(Data, row * col, num_feature),dr,num_PC);
            XDRvector = XDR;
            minZ = min(XDRvector);
            maxZ = max(XDRvector);
            XDRvector = bsxfun(@minus, XDRvector, minZ);
            XDRvector = bsxfun(@rdivide, XDRvector, maxZ-minZ);
            
            %Step 2: ����׻����� XWhiten
            XDR_cov = cov(XDR);
            [U S V] = svd(XDR_cov);
            whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
            XDR = XDR * whiten_matrix;
            XDR = bsxfun(@rdivide,bsxfun(@minus,XDR,mean(XDR,1)),std(XDR,0,1)+epsilon);
            XDR = reshape(XDR,row,col,num_PC);
            X_extension = MirrowCut(XDR,win_inter);
            %�Ӱ׻�����XWhiten����ȡk�������.
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
    
    
    
    X_joint = [];
    for i=1:Layernum
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
    
    %����SVM����
    best_c = 1024;
    best_g = 2^-6;
    svm_option = horzcat('-c',' ',num2str(best_c),' -g',' ',num2str(best_g));
    
    %ѵ��SVM
    model = libsvm_svmtrain(y_train,X_train,svm_option);
    %Ԥ��
    [predict_label, ~, ~] = svmpredict(y_test, X_test, model);
    %����OA��Kappa���Լ�ÿ������׼ȷ��
    [OA, Kappa, producerA] = CalAccuracy(predict_label,y_test);
    
    %����OA��Kappa
    OA_set(ii)=OA;
    Kappa_set(ii)=Kappa;
    
    
    [labels, accuracy, dec_values] = svmpredict(Label, X_joint, model);
    X_result = drawresult(labels,row,col, 2);
    
    %���Ԥ��ͼ��
    str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
    str1=strcat('KSC_',dr,'_svm_',ActivationFunction);
    str2='.png';
    save_path=[str0,str1,str2];
    imwrite(X_result,save_path);
end

%����OA��Kappa�ľ�ֵ
OA_average=mean(OA_set);
Kappa_average=mean(Kappa_set);
%���OA��Kappa��time
disp(['������:SVM']);
disp(['��ά����:',dr]);
disp(['��ά����:',num2str(num_PC)]);
disp(['�������:',num2str(Layernum)]);
disp(['�����:',ActivationFunction]);
disp(['OA= ',num2str(OA_average)]);
disp(['Kappa=',num2str(Kappa_average)]);
disp(['time=',num2str(toc/repeat)]);


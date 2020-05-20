%%绘制Indian Pines 地面真值
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
clear a;
X_result = drawresult(Label,row,col, 2);

%输出真值图像
str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
str1='Indian_Pines_GroundTruth';
str2='.png';
save_path=[str0,str1,str2];
imwrite(X_result,save_path);
%%


%%绘制PaviaU 地面真值
% 加载数据
data_name='PaviaU.mat';
data_gt='PaviaU_gt.mat';
addpath(genpath('utils'))
addpath(genpath('dataset'))
a=load(data_name);
Data = a.paviaU;
[row,col,num_feature] = size(Data);
a=load(data_gt);
Label = reshape(double(a.paviaU_gt),row*col,1);
clear a;
X_result = drawresult(Label,row,col, 2);

%输出真值图像
str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
str1='PaviaU_GroundTruth';
str2='.png';
save_path=[str0,str1,str2];
imwrite(X_result,save_path);
%%



%%绘制KSC 地面真值
% 加载数据
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
X_result = drawresult(Label,row,col, 2);

%输出真值图像
str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
str1='KSC_GroundTruth';
str2='.png';
save_path=[str0,str1,str2];
imwrite(X_result,save_path);
%%


%%绘制Salinas 地面真值
% 加载数据
data_name='Salinas.mat';
data_gt='Salinas_gt.mat';
addpath(genpath('utils'))
addpath(genpath('dataset'))
a=load(data_name);
Data = a.salinas;
[row,col,num_feature] = size(Data);
a=load(data_gt);%ground truth
Label = reshape(double(a.salinas_gt),row*col,1);
num_class = max(Label(:));
clear a;
X_result = drawresult(Label,row,col, 2);

%输出真值图像
str0='C:\Documents\graduation_project\HSIC_RPNet\figure\';
str1='Salinas_GroundTruth';
str2='.png';
save_path=[str0,str1,str2];
imwrite(X_result,save_path);
%%
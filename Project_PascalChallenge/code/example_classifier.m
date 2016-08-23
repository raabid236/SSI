function example_classifier
clc
clear all
close all
run C:\Vlfeat\vlfeat-0.9.20-bin.tar\vlfeat-0.9.20-bin\vlfeat-0.9.20\toolbox\vl_setup

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

addpath c:\Users\Gourab\Documents\prtools4.2.5\prtools
% bag of words
tree = bagOfWords(VOCopts,500);

% train and test classifier for each class
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
%     classifier = train(VOCopts,cls,tree);      % train classifier
%     test(VOCopts,cls,classifier,tree);         % test classifier
    [svmw,svmb] = svmtrain(VOCopts,cls,tree,0.00001,-1);
    svmtest(VOCopts,cls,svmw,svmb,tree,0.00001,-1)
%     w = trainboost(VOCopts,cls,tree);
%     testboost(VOCopts,cls,w,tree)
    [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   % compute and display ROC
    
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end

% Bag of Words
function tree = bagOfWords(VOCopts,numClusters)
data = [];
for i=1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    [ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    % extract features for each image
    counter = 0;
    for j=1:length(ids)
        % display progress
        if counter == 6
            break;
        end
        if classifier.gt(j) == -1
            continue;
        end
        counter = counter + 1;
        
        try
            % try to load features
            load(sprintf(VOCopts.exfdpath,ids{j}),'fd');
            tree = [];
            return
        catch
            % compute and save features
            I=imread(sprintf(VOCopts.imgpath,ids{j}));
            fd =extractsiftcolortexturefd(VOCopts,I);
            data = [data,fd];
        end
    end
end
size(data)
[tree,~] = vl_hikmeans(uint8(data),numClusters,10);
disp('Dictionary built')

% train classifier
function classifier = train(VOCopts,cls,tree)

% load 'train' image set for class
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');

% extract features for each image
classifier.FD=zeros(0,length(ids));
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d=extractsiftcolorfd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    classifier.FD(1:length(fd),i)=fd;
end


% SVM Train
function [svmw,svmb] = svmtrain(VOCopts,cls,tree,lambda,kernel)

% load 'train' image set for class
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
% extract features for each image
classifier.FD=zeros(0,length(ids));
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d=extractsiftcolortexturefd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    classifier.FD(1:length(fd),i)=fd;
end
if (kernel ~= -1)
    hom.kernel = 'KChi2';
    hom.order = kernel;
    dataset = vl_svmdataset(classifier.FD, 'homkermap', hom);
else
    dataset = classifier.FD;
end
[svmw svmb ~] = vl_svmtrain(dataset, double(classifier.gt), lambda);

% Boost Train
function w = trainboost(VOCopts,cls,tree)

% load 'train' image set for class
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
% extract features for each image
classifier.FD=zeros(0,length(ids));
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d=extracthogfd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    classifier.FD(1:length(fd),i)=fd;
end
data = dataset(classifier.FD',classifier.gt);
data = setprior(data,getprior(data));
w = adaboostc(data,perlc([],300));

% run classifier on test images
function test(VOCopts,cls,classifier,tree)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

% classify each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d=extractsiftcolorfd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification
    c=classify(VOCopts,classifier,fd);
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file
fclose(fid);


% run SVM classifier on test images
function svmtest(VOCopts,cls,svmw,svmb,tree,lambda,kernel)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

% classify each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d =extractsiftcolortexturefd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification
    if (kernel ~= -1)
        hom.kernel = 'KChi2';
        hom.order = kernel;
        dataset = vl_svmdataset(fd, 'homkermap', hom);
        [~,~,~,c] = vl_svmtrain(dataset, 1000, lambda, 'model', svmw, 'bias', svmb, 'solver', 'none');
    else
        c=(svmw'*fd + svmb);
    end
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file
fclose(fid);

% run SVM classifier on test images
function testboost(VOCopts,cls,w,tree)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

testdata = [];
% classify each image
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        d =extracthogfd(VOCopts,I);
        path = vl_hikmeanspush(tree,uint8(d));
        fd = vl_hikmeanshist(tree,path);
        fd = double(fd(2:end)/fd(1));
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification
    testdata = [testdata; fd'];
    
    % write to results file
    %fprintf(fid,'%s %f\n',ids{i},c);
end

data = dataset(testdata,gt);
data = setprior(data,getprior(data));
fclass=data*w*classc;
temp=+fclass;
for i=1:length(ids)
    fprintf(fid,'%s %f\n',ids{i},temp(i,2));
end

% close results file
fclose(fid);


% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts,I)

fd = [];
[nr,nc,nz] = size(I);
for i=1:10,
	for j=1:10,
		dv = I(floor(1+(i-1)*nr/10):floor(i*nr/10),floor(1+(j-1)*nc/10):floor(j*nc/10),:);
		fd = [fd;sum(sum(double(dv)))/(size(dv,1)*size(dv,2))];
%fd=squeeze(sum(sum(double(I)))/(size(I,1)*size(I,2)));
	end
end
fd = fd(:);

% SIFT feature extractor:
function fd = extractsiftfd(VOCopts,I)
I = single(rgb2gray(I));
[~,desc] = vl_sift(I);
% write a loop to extract color from the patch around all points of f
% and concatenate to desc
fd=double(desc);

% SIFT feature and color extractor:
function fd = extractsiftcolorfd(VOCopts,I)
[m,n,~] = size(I);
[~,desc1] = vl_sift(single(reshape(I(:,:,1),m,n)));
[~,desc2] = vl_sift(single(reshape(I(:,:,2),m,n)));
[~,desc3] = vl_sift(single(reshape(I(:,:,3),m,n)));
fd = double([desc1,desc2,desc3]);

% SIFT feature, color and texture extractor:
function fd = extractsiftcolortexturefd(VOCopts,I)
[m,n,~] = size(I);
[~,desc1] = vl_sift(single(reshape(I(:,:,1),m,n)));
[~,desc2] = vl_sift(single(reshape(I(:,:,2),m,n)));
[~,desc3] = vl_sift(single(reshape(I(:,:,3),m,n)));
se=strel('square',7);
[~,desc4] = vl_sift(single(stdfilt(rgb2gray(I),getnhood(se))));
fd = double([desc1,desc2,desc3,desc4]);


% HOG feature extractor:
function fd = extracthogfd(VOCopts,I)
hog = vl_hog(single(I),8);
[m,n,~] = size(hog);
fd = [];
for i = 1:m
    for j = 1:n
        temp = reshape(double(hog(i,j,:)),31,1);
        fd = [fd, 1024*temp ];
    end
end

% trivial classifier: compute ratio of L2 distance betweeen
% nearest positive (class) feature vector and nearest negative (non-class)
% feature vector
function c = classify(VOCopts,classifier,fd)

d=sum(fd.*fd)+sum(classifier.FD.*classifier.FD)-2*fd'*classifier.FD;
dp=min(d(classifier.gt>0));
dn=min(d(classifier.gt<0));
c=dn/(dp+eps);
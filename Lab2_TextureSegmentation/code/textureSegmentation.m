% Script to build local features based on texture 
% and run region growing segmentation

clc
clear all
close all

% Image Filename
filename = 'hand2.tif';


% Parameters
patch_size = 5;
D = 1;
offset = [D D];
numLevels = 4;
dimensions = 4;


I = imread(filename);
figure()
imshow(I)
title('Input Image')

Ig = rgb2gray(I);
[m,n] = size(Ig);
h = fspecial('gaussian',5,4);
Ig = conv2(double(Ig), h, 'same');           % Gaussian smoothing

val = round(floor(patch_size/2));
Ig = padarray(Ig,[val val],'symmetric');
[newm,newn] = size(Ig);

% Local features for each pixel
Itext = zeros(m,n,dimensions);
for i = 1+val:newm-val
    for j = 1+val:newn-val
        Itemp = Ig(i - val: i + val, j - val: j + val);    % Extract neighborhood
        % Compute GLCM and properties
        glcm = graycomatrix(uint8(Itemp),'Offset',offset,'NumLevels',numLevels,'Symmetric',true);
        stats = graycoprops(glcm,{'contrast','energy','homogeneity'});
        % Compute Entropy
        [gm,gn] = size(glcm);
        entropy = 0;
        for ki = 1:gm
            for kj = 1:gn
                temp = (glcm(ki,kj)/sum(glcm(:)));
                entropy = entropy + temp*log10(temp+0.001);
            end
        end
        
        % Store properties in matrix
        Itext(i-val,j-val,1) = stats.Contrast;
        Itext(i-val,j-val,2) = stats.Energy;
        Itext(i-val,j-val,3) = stats.Homogeneity;
        Itext(i-val,j-val,4) = entropy;
    end
end

% Display Texture images
temp = Itext(:,:,1);
Itext(:,:,1) = (Itext(:,:,1)-min(temp(:))/(max(temp(:))-min(temp(:))));
figure()
imshow(Itext(:,:,1))
title('Contrast Image')

temp = Itext(:,:,2);
Itext(:,:,2) = (Itext(:,:,2)-min(temp(:))/(max(temp(:))-min(temp(:))));
figure()
imshow(Itext(:,:,2))
title('Energy Image')

temp = Itext(:,:,3);
Itext(:,:,3) = (Itext(:,:,3)-min(temp(:))/(max(temp(:))-min(temp(:))));
figure()
imshow(Itext(:,:,3))
title('Homogeneity Image')

temp = Itext(:,:,4);
Itext(:,:,4) = (Itext(:,:,4)-min(temp(:))/(max(temp(:))-min(temp(:))));
figure()
imshow(Itext(:,:,4))
title('Entropy Image')

% Segmentation

% Region Growing with only RGB
feature_dimensions = 3;
data = zeros(m,n,feature_dimensions);
for i = 1:m
    for j = 1:n
        data(i,j,1) = I(i,j,1);
        data(i,j,2) = I(i,j,2);
        data(i,j,3) = I(i,j,3);
    end
end
outputrg = regiongrowing(data,35,4);

figure()
imshow(1-((double(outputrg)-min(outputrg(:)))/(max(outputrg(:)-min(outputrg(:))))))
title('Region Growing with color - Segmented Image')

% Region Growing with RGB and texture features

% Build features for each pixel
feature_dimensions = 5;
data = zeros(m,n,feature_dimensions);
for i = 1:m
    for j = 1:n
        data(i,j,1) = I(i,j,1)/255;   % R
        data(i,j,2) = I(i,j,2)/255;   % G
        data(i,j,3) = I(i,j,3)/255;   % B
        data(i,j,4) = Itext(i,j,1);   % Contrast
        data(i,j,5) = Itext(i,j,4);   % Entropy
    end
end
outputrg = regiongrowing(data,0.8,8);

figure()
imshow(1-((double(outputrg)-min(outputrg(:)))/(max(outputrg(:)-min(outputrg(:))))))
title('Region Growing with Color and Texture - Segmented Image')



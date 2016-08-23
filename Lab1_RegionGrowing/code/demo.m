% Segmentation Algorithms
% Gourab Ghosh Roy and Raabid Hussain

clc
clear all
close all

% Region Growing

% Parameters - Image (RGB or grayscale) filename, Value of threshold, Size of neighborhood (4 or 8)
% Output - Image with pixel values being the region number 
outputrg = regiongrowing('color.tif',25,8);


% Fuzzy C Means

% Parameters - Image filename (RGB or grayscale), Number of clusters
% Output - Image with pixel values being the region number 
%outputfcm = fuzzycm('image2.png',4);



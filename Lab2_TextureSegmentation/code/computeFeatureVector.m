function v = computeFeatureVector(A)
%
% Describe an image A using intensity features.
%   A is the image
%   v is a 1xN vector, being N the number of features used to describe the
% image
%

if size(A,3) > 1,
	A = rgb2gray(A);
end

v = mean(A);

function v = computeTextureFeatureVector(A)
%
% Describe an image A using texture features.
%   A is the image
%   v is a 1xN vector, being N the number of features used to describe the
% image
%


Ig = rgb2gray(A);
numLevels = 16;

v=[];
for D = 1:4:13
    k = [0 D; -D D; -D 0; -D -D];  % offset values
    for j = 1:4
        offset = k(j,:);
        % Compute GLCM and properties
        glcm = graycomatrix(uint8(Ig),'Offset',offset,'NumLevels',numLevels,'Symmetric',true);
        stats = graycoprops(glcm,{'contrast','energy','homogeneity','correlation'});
        % Compute Entropy
        [gm,gn] = size(glcm);
        entropy = 0;
        for ki = 1:gm
            for kj = 1:gn
                temp = (glcm(ki,kj)/sum(glcm(:)));
                entropy = entropy + temp*log10(temp+0.001);
            end
        end
        % Concatenate features together
        v = [v,stats.Contrast,stats.Homogeneity,stats.Correlation];
    end
end
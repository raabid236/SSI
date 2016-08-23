function [ markedI ] = fuzzycm(filename,num_cluster)

    I = imread(filename);
    figure()
    imshow(I);
    title('Fuzzy C Means - Input Image')
    I = double(I);
    d = size(I,3);
    if d == 1
        [m,n] = size(I);
        data = reshape(I,m*n,1);
    else
        [m,n,~] = size(I);
        data = [reshape(I(:,:,1),m*n,1),reshape(I(:,:,2),m*n,1),reshape(I(:,:,3),m*n,1)];
    end

    options = [NaN NaN NaN 0];
    [~,U,~] = fcm(data,num_cluster,options);

    markedI = zeros(m,n);
    for j = 1:n
        for i = 1:m
            for k = 1:num_cluster
                [~, ind] = max(U(:,m*(j-1)+i));
                markedI(i,j) = ind;
            end
        end
    end

    figure()
    imshow(1-((double(markedI)-min(markedI(:)))/max(markedI(:))))
    title('Fuzzy C Means - Segmented Image')


end


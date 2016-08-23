function [ markedI ] = regiongrowing(I,thr,neighborhood)
    % Code for Region Growing Segmentation
    
    d = size(I,3);
    if d == 1
        [m,n] = size(I);
    else
        [m,n,~] = size(I);
    end

    I = double(I);
    markedI = zeros(m,n);
    numregion = 0;
    q = [];

    for i = 1:m
        for j = 1:n
            if (markedI(i,j) ~= 0)         % Check if pixel already belongs to another region
                continue;
            end
            if isempty(q)                                      % Start a new region when the current queue is empty
                numregion = numregion + 1;                              
                if (markedI(i,j) == 0)                         
                    markedI(i,j) = numregion;                  % Start the region with the current pixel
                    if d == 1
                        meanR = I(i,j);                             
                    else                                       % Update region mean accordingly  
                        meanR = [];
                        for k = 1: d
                            meanR = [meanR, I(i,j,k)];
                        end
                    end
                    qpixel = 1;
                    % Add neighbors of current pixel to the dynamic queue
                    % if they do not belong to any other region
                    if ((i+1)<=m) && (markedI(i+1,j) == 0)
                        testp = (i)*n + j;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((i-1)>=1) && (markedI(i-1,j) == 0)
                        testp = (i-2)*n + j;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((j+1)<=n) && (markedI(i,j+1) == 0)
                        testp = (i-1)*n + j+1;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((j-1)>=1) && (markedI(i,j-1) == 0)
                        testp = (i-1)*n + j-1;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if (neighborhood == 8)           % 4 more neighbors in case of A8 connectivity
                        if ((i+1)<=m) && ((j+1)<=n) && (markedI(i+1,j+1) == 0)
                            testp = (i)*n + j+1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((i-1)>=1) && ((j+1)<=n) && (markedI(i-1,j+1) == 0)
                            testp = (i-2)*n + j+1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((i+1)<=m) && ((j-1)>=1) && (markedI(i+1,j-1) == 0)
                            testp = (i)*n + j-1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((i-1)>=1) && ((j-1)>=1) && (markedI(i-1,j-1) == 0)
                            testp = (i-2)*n + j-1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end                    
                    end
                end
            end
            % Explore till the queue for current region is empty
            while (~isempty(q))
                % Select the first pixel from queue
                worki = ceil(q(1) / n);
                workj = mod(q(1),n);
                if (worki > m)
                    worki = m;
                end
                if (workj == 0)
                    workj = n;
                end
                % Calculate Euclidean distance between region mean and current pixel 
                if d == 1
                    dist = norm(meanR - I(worki,workj));
                else
                    temp = [];
                    for k = 1:d
                        temp = [temp, I(worki,workj,k)];
                    end
                    dist = norm(meanR - temp);
                end
                % Check if less than threshold
                if (dist <= thr)
                    markedI(worki,workj) = numregion;   % Assign pixel to the current region
                    % Update Region statistics
                    if d == 1
                        meanR = (meanR*qpixel + I(worki,workj))/(qpixel+1);
                    else
                        meanR = (meanR*qpixel + temp)/(qpixel+1);
                    end
                               
                    qpixel = qpixel + 1;          % update number of pixels in current region
                    % Add neighbors of current pixel to the queue
                    % if they do not belong to any other region
                    if ((worki+1)<=m) && (markedI(worki+1,workj) == 0)
                        testp = (worki)*n + workj;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((worki-1)>=1) && (markedI(worki-1,workj) == 0)
                        testp = (worki-2)*n + workj;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((workj+1)<=n) && (markedI(worki,workj+1) == 0)
                        testp = (worki-1)*n + workj+1;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if ((workj-1)>=1) && (markedI(worki,workj-1) == 0)
                        testp = (worki-1)*n + workj-1;
                        if (any(q==testp) == 0)
                            q = [q;testp];
                        end
                    end
                    if (neighborhood == 8)
                        if ((worki+1)<=m) && ((workj+1)<=n) && (markedI(worki+1,workj+1) == 0)
                            testp = (worki)*n + workj+1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((worki-1)>=1) && ((workj+1)<=n) && (markedI(worki-1,workj+1) == 0)
                            testp = (worki-2)*n + workj+1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((worki+1)<=m) && ((workj-1)>=1) && (markedI(worki+1,workj-1) == 0)
                            testp = (worki)*n + workj-1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end
                        if ((worki-1)>=1) && ((workj-1)>=1) && (markedI(worki-1,workj-1) == 0)
                            testp = (worki-2)*n + workj-1;
                            if (any(q==testp) == 0)
                                q = [q;testp];
                            end
                        end                    
                    end
                end

                % Take the current pixel out of the queue
                q = q(2:end);
            end        
        end
    end
    
    
end


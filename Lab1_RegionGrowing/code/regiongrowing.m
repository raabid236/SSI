function [ markedI ] = regiongrowing(filename,thr,neighborhood)

    I = imread(filename);

    d = size(I,3);
    if d == 1
        [m,n] = size(I);
    else
        [m,n,~] = size(I);
    end
    figure()
    imshow(I);
    title('Region Growing - Input Image')

    I = double(I);
    markedI = zeros(m,n);
    numregion = 0;
    q = [];

    for i = 1:m
        for j = 1:n
            if (markedI(i,j) ~= 0)
                continue;
            end
            if isempty(q)
                numregion = numregion + 1;
                if (markedI(i,j) == 0)
                    markedI(i,j) = numregion;
                    if d == 1
                        meanR = I(i,j);
                    else
                        meanR = [I(i,j,1) I(i,j,2) I(i,j,3)];
                    end
                    qpixel = 1;
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
                    if (neighborhood == 8)
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
            while (~isempty(q))
                worki = ceil(q(1) / n);
                workj = mod(q(1),n);
                if (worki > m)
                    worki = m;
                end
                if (workj == 0)
                    workj = n;
                end

                if d == 1
                    dist = norm(meanR - I(worki,workj));
                else
                    dist = norm(meanR - [I(worki,workj,1) I(worki,workj,2) I(worki,workj,3)]);
                end
                if (dist <= thr)
                    markedI(worki,workj) = numregion;
                    if d == 1
                        meanR = (meanR*qpixel + I(worki,workj))/(qpixel+1);
                    else
                        temp = [I(worki,workj,1) I(worki,workj,2) I(worki,workj,3)];
                        meanR = (meanR*qpixel + temp)/(qpixel+1);
                    end

                    qpixel = qpixel + 1;
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


                q = q(2:end);
            end        
        end
    end

    figure()
    imshow((1-((double(markedI)-min(markedI(:)))/max(markedI(:)))))
    title('Region Growing - Segmented Image')
    imwrite((1-((double(markedI)-min(markedI(:)))/max(markedI(:)))),'reg2.png');
end


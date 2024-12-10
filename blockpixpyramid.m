function [C,Cloc,Pyramid,parents]=blockpixpyramid(A,oldC,alphat,parents,totalpixels,Img)
    n = size(A,2)
    C = 1;
    Cloc = 1;
    Pyramid{1} = 1;
    %Pyramid{totalpixels}= 0;
    ASum=alphat*sum(A,1);
    for i = 2:n
        if max(A(i,Cloc)) < ASum(i) % alphat*sum(A(i,:))
            Cloc = [Cloc,i];
            C = [C,oldC(i)];
            %Pyramid{oldC(i)} = oldC(i);
            Pyramid{i} = oldC(i);
            parents{oldC(i)} = 0;
        else
%             A(i,:)
%             oldC(find(A(i,:)>.9))
%             pause 

%             Pyramid{oldC(i)} = oldC(find(A(i,:)>.9)); %only for childs
%             parents{oldC(i)} = Pyramid{oldC(i)};
            
            Pyramid{i} = oldC(find(A(i,:)./max(A(i,:))>.2)); %only for childs
            parents{oldC(i)} = Pyramid{i};
        end
        if mod(i,1000)==0
            disp(['Block Pixel iteration: ',num2str(i)])
        end
    end
%     
%     K = size(C,2);
end
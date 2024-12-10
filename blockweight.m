function [P,newA,Vol] = blockweight(W,C,Cloc,Vol)
    
    size(C)
    size(Cloc)
    P = sparse(zeros(size(W,2),size(C,2)));
    k = 1;
    %ASum=sum(A,1);
    denomzero = [];

    for i = 1:size(W,2)
        if ismember(i,Cloc)
            P(i,k) = 1;
            k = k+1;
        else
            denom = sum(W(i,Cloc));
            if denom ==0
              %  error('W values are too small for the chosen coarse nodes.')
%                 denomzero = [denomzero,i];
                 denom=1;
            end
            %denom=1;
            [Cloc,idperm]=sort(Cloc);
            P(i,:) = W(i,Cloc)./denom;
        end
        
        if mod(i,100)==0
            disp(['Block weight iteration: ',num2str(i)])
        end
    end
    
    newA = (P'*W)*P;
 
    Vol = Vol*P;
end
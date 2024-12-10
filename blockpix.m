function [C,K,Cloc]=blockpix(A,oldC,alphat)
    n = size(A,2)
    C = 1;
    Cloc = 1;
    ASum=alphat*sum(A,1);
    for i = 2:n
        if max(A(i,Cloc)) < ASum(i) % alphat*sum(A(i,:))
            Cloc = [Cloc,i];
            C = [C,oldC(i)];
        end
        if mod(i,1000)==0
            i
        end
    end
    
    K = size(C,2);
end
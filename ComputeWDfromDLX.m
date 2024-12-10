function [W,D] = ComputeWDfromDLX(x,sigmaX)
n=size(x,1);
for i =1:n
    for j =1:n
        if i==j
            W(i,j)=0;
        else
            W(i,j) = exp(-(norm(x(i,:)-x(j,:))/(sigmaX^2)));
        end     
    end
end
d = sum(W,2);
D = diag(d)
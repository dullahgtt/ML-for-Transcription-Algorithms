function [G,modW]=modifyWvec(x,P,C,newW,mu)
    K=size(C,2);
    G=zeros(1,K);
    Ptemp = P{1};
    for i=2:size(P,2)
        Ptemp = Ptemp*P{i};
    end
    %% Compute the aggregate 
    for k =1:K
        k
        num = sum(full(Ptemp(:,k))'*x',2);
        denom = sum(full(Ptemp(:,k)));
        G(k,:)=num/denom;
    end
    
    modW = zeros(K,K);
    for k=1:K
        k
        for l=1:K
            modW(k,l)= newW(k,l)*exp(-mu*norm(G(k,:)-G(l,:),2));
        end
        % Normalize the Modified Affinity Matrix A
        % modW(k,:) = modW(k,:)./max(modW(k,:));
    end
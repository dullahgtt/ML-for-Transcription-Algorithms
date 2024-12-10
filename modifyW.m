function [G,modW]=modifyW(Img,P,C,newW,mu)
    K=size(C,2);
    G=zeros(1,K);
    [nr,nc]=size(Img);
    cImg1 = reshape(Img,nr*nc,1);
    Ptemp = P{1};
    for i=2:size(P,2)
        Ptemp = Ptemp*P{i};
    end
    
    %% Compute the aggregate 
    for k =1:K
        k
        num = sum(Ptemp(:,k).*cImg1);
        denom = sum(Ptemp(:,k));
        G(k)=num/denom;
    end
    
    modW = zeros(K,K);
    for k=1:K
        k
        for l=1:K
            modW(k,l)= newW(k,l)*exp(-mu*abs(G(k)-G(l)));
        end
        % Normalize the Modified Affinity Matrix A
        % modW(k,:) = modW(k,:)./max(modW(k,:));
    end
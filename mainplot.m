clear all
close all 

for i =8:9
load(['Bed00',num2str(i)])
filename = ['Bed00',num2str(i)];
m=800000;

nlen = linspace(1,size(CPUtime,2),size(CPUtime,2))*m/fs;
figure
plot(nlen,CPUtime)
legend('CSS','kmeans','SC','PCA+kmeans')
xlabel('length of audio file in seconds')
ylabel('CPUtime in seconds')
title(['CPU time - ', filename])
saveas(gcf,[filename,'CPU.png'])
close all

nlen = linspace(1,size(SpeakerErrorRate,2),size(SpeakerErrorRate,2))*m/fs;
figure
plot(nlen,SpeakerErrorRate)
legend('CSS','kmeans','SC','PCA+kmeans')
xlabel('length of audio file in seconds')
ylabel('Error Rate')
title(['Error Rate - ', filename])
saveas(gcf,[filename,'Error.png'])
close all
end


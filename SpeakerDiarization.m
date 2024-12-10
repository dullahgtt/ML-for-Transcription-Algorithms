%% Speaker Diarization Using x-vectors
% Speaker diarization is the process of partitioning an audio signal into segments
% according to speaker identity. It answers the question "who spoke when" without
% prior knowledge of the speakers and, depending on the application, without prior
% knowledge of the number of speakers.
%
% Speaker diarization has many applications, including: enhancing speech transcription
% by structuring text according to active speaker, video captioning, content retrieval
% (_what did Jane say?_) and speaker counting (_how many speakers were present
% in the meeting?_).
%
% In this example, you perform speaker diarization using a pretrained x-vector
% system [1] to characterize regions of audio and agglomerative hierarchical clustering
% (AHC) to group similar regions of audio [2]. To see how the x-vector system
% was defined and trained, see <docid:audio_ug#mw_e0ae8946-7ecc-499a-bdfb-caeedd325bc3
% Speaker Recognition Using x-vectors>.
%% Download Pretrained Speaker Diarization System
% Download the pretrained speaker diarization system and supporting files. The
% total size is approximately 22 MB.

clear all
close all

downloadFolder = matlab.internal.examples.downloadSupportFile("audio","SpeakerDiarization.zip");
dataFolder = tempdir;
unzip(downloadFolder,dataFolder)
netFolder = fullfile(dataFolder,"SpeakerDiarization");

addpath(netFolder)

%% Load from MATLAB
 %[audioIn,fs] = audioread("exampleconversation.flac");
 %pause
 %load("exampleconversationlabels.mat")
% audioIn = audioIn./max(abs(audioIn));
% %sound(audioIn,fs)
% t = (0:size(audioIn,1)-1)/fs;


%% Load the data from ICSI
filename = 'Bmr015';
[audioIn0,fs] = audioread([filename,'.wav']);
audiolength = size(audioIn0,1);
nmax = floor(audiolength/800000);
totaltime=tic;
for n=1:nmax
    nlooptime = tic ;
audioIn = audioIn0(1:n*800000); % Change the second number to run a larger sample
t = (0:size(audioIn,1)-1)/fs;
%% Load the ground truth
text = fileread([filename,'.mrt']);
k = strfind(text,'<Segment');
participant=[];
for i=1:size(k,2)
    StartTime(i) = str2num(text(k(i)+20:k(i)+24));
    if StartTime<10
        EndTime(i) = str2num(text(k(i)+36:k(i)+40));
    end
    if StartTime(i)>=10 && StartTime(i)<100
        StartTime(i) = str2num(text(k(i)+20:k(i)+25));
        EndTime(i) = str2num(text(k(i)+37:k(i)+42));
    end 
    
    if StartTime(i)>=100 && StartTime(i)<1000
        StartTime(i) = str2num(text(k(i)+20:k(i)+26));
        EndTime(i) = str2num(text(k(i)+38:k(i)+44));
    end 
    
    if StartTime(i)>=1000
        StartTime(i) = str2num(text(k(i)+20:k(i)+27));
        EndTime(i) = str2num(text(k(i)+39:k(i)+46));
    end

    temptext = text(k(i):k(i)+74);
    tempk = strfind(temptext,'Participant=');
    if isempty(tempk)
        participant{i} = 'SP0';
    else
        participant{i} = temptext(tempk+13:tempk+17);
    end
end
clear text

% figure(1)
% set(gcf, 'Position',  [100, 100, 1500, 200])
% plot(t,audioIn)
% xlabel("Time (s)")
% ylabel("Amplitude")
% axis tight


%% Extract x-vectors
% In this example, you used a pretrained x-vector system based on [1]. To see
% how the x-vector system was defined and trained, see <docid:audio_ug#mw_e0ae8946-7ecc-499a-bdfb-caeedd325bc3
% Speaker Recognition Using x-vectors>.
%% Load Pretrained x-Vector System
% Load the lightweight pretrained x-vector system. The x-vector system consists
% of:
%%
% * |afe| - an <docid:audio_ref#mw_b56cd7dc-af31-4da4-a43e-b13debc30322 |audioFeatureExtractor|>
% object to extract mel frequency cepstral coefficients (MFCCs).
% * |factors| - a struct containing the mean and standard deviation of MFCCs
% determined from a representative data set. These factors are used to standardize
% the MFCCs.
% * |dlnet| - a trained |dlnetwork|. The network is used to extract x-vectors
% from the MFCCs.
% * |projMat| - a trained projection matrix to reduce the dimensionality of
% x-vectors.
% * |plda| - a trained PLDA model for scoring x-vectors.

xvecsys = load("xvectorSystem.mat");
%% Extract Standardized Acoustic Features
% Extract standardized MFCC features from the audio data. View the feature distributions
% to confirm that the standardization factors learned from a separate data set
% approximately standardize the features derived in this example. A standard distribution
% has a mean of zero and a standard deviation of 1.

features = single((extract(xvecsys.afe,audioIn)-xvecsys.factors.Mean')./xvecsys.factors.STD');

% figure(2)
% histogram(features)
% xlabel("Standardized MFCC")



%% Extract x-Vectors
% Each acoustic feature vector represents approximately 0.01 seconds of audio
% data. Group the features into approximately 2 second segments with 0.1 second
% hops between segments.

featureVectorHopDur = (numel(xvecsys.afe.Window) - xvecsys.afe.OverlapLength)/xvecsys.afe.SampleRate;

segmentDur = 2; %default is 2
segmentHopDur = 0.5; %default is 0.1

segmentLength = round(segmentDur/featureVectorHopDur);
segmentHop = round(segmentHopDur/featureVectorHopDur);

idx = 1:segmentLength;
featuresSegmented = [];
while idx(end) < size(features,1)
    featuresSegmented = cat(3,featuresSegmented,features(idx,:));
    idx = idx + segmentHop;
end
%%
% Extract x-vectors from each segment. x-vectors correspond to the output from
% the first fully-connected layer in the x-vector model trained in <docid:audio_ug#mw_e0ae8946-7ecc-499a-bdfb-caeedd325bc3
% Speaker Recognition Using x-vectors>. The first fully-connected layer is the
% first segment-level layer after statistics are calculated for the time-dilated
% frame-level layers. Visualize the x-vectors over time.

xvecs = zeros(512,size(featuresSegmented,3));
for sample = 1:size(featuresSegmented,3)
    dlX{sample} = dlarray(featuresSegmented(:,:,sample),"TCB");
    xvecs(:,sample) = predict(xvecsys.dlnet,dlX{sample},Outputs="fc_1");
end

% figure(3)
% surf(xvecs',EdgeColor="none")
% view([90,-90])
% axis([1 size(xvecs,1) 1 size(xvecs,2)])
% xlabel("Features")
% ylabel("Segment")



%%
% Apply the pretrained linear discriminant analysis (LDA) projection matrix
% to reduce the dimensionality of the x-vectors and then visualize the x-vectors
% over time.

x = xvecsys.projMat*xvecs;

% figure(4)
% surf(x',EdgeColor="none")
% view([90,-90])
% axis([1 size(x,1) 1 size(x,2)])
% xlabel("Features")
% ylabel("Segment")



%% Cluster x-vectors
% An x-vector system learns to extract compact representations (x-vectors) of
% speakers. Cluster the x-vectors to group similar regions of audio using either
% agglomerative hierarchical clustering (<docid:stats_ug#bsso096-1 |clusterdata|>)
% or k-means clustering (<docid:stats_ug#buefs04 |kmeans|>). [2] suggests using
% agglomerative heirarchical clustering with PLDA scoring as the distance measurement.
% K-means clustering using a cosine similarity score is also commonly used. Assume
% prior knowledge of the number of speakers in the audio. Set the maximum
% clusters to the number of known speakers + 1 so that the background is clustered
% independently.


%% Use a known method to cluster
%knownNumberOfSpeakers = numel(unique(groundTruth.Label));
knownNumberOfSpeakers = numel(unique(participant));
maxclusters = knownNumberOfSpeakers+1;

totalmethods =5;
name = "";
for i=2:totalmethods
    clusterMethod = i;
    clear T
    switch clusterMethod
        %case "agglomerative - PLDA scoring"
        case 1
            tic
            name = "PLDA scoring";
            T = clusterdata(x',Criterion="distance",distance=@(a,b)helperPLDAScorer(a,b,xvecsys.plda),linkage="average",maxclust=maxclusters);
            
            CPUtime(i-1,n) = toc;
        %case "agglomerative - CSS scoring"
        case 2
            tic
            name = "CSS scoring";
            T = clusterdata(x',Criterion="distance",distance="cosine",linkage="average",maxclust=maxclusters);
            
            CPUtime(i-1,n) = toc;
        %case "kmeans - CSS scoring"
        case 3
            tic
            name = "K means";
            T = kmeans(x',maxclusters);%,'distance','cosine');
            CPUtime(i-1,n) = toc;
        %case "spectral clustering"
        case 4
            tic
            name = "Spectral clustering";
            T = spectralcluster(x',maxclusters);%,Criterion="distance");
            CPUtime(i-1,n) = toc;
        %case "principal component analysis"
        case 5
            tic
            name = "PCA";
            [coeff, score, latent] = pca(x');
            P = coeff(:, 1:maxclusters);
            T1 = x' * P;
            T = kmeans(T1,maxclusters);
            CPUtime(i-1,n) = toc;
    end

    % %% New Clustering Method
    % sigmaX = .1;
    % [W{1},D]= ComputeWDfromDLX(x',sigmaX);
    % 
    % %% Initialize location mapping u and nodes C
    % n = size(W{1},2);
    % u.list = linspace(1,n,n);
    % %u.loc = reshape(linspace(1,n,n),size(Img,1),size(Img,2));
    % C{1}=u.list;
    % 
    % %% STEP I: Obtain Coarsening, block pix (input: A, u.list,alphat)
    % level = 10;
    % alphat = .1;
    % alpha = 2;
    % modW{1}=W{1};
    % %Athresh = modW{1};
    % %cImg=reshape(Img,n,1);
    % Cthresh = C{1};
    % %% Segmentation by weighted aggregation (coarsening)
    % for i =1:level
    %     clear Cloc
    %     tic
    %     [C{i+1},Cloc] = blockvec(modW{i},C{i},C{i},alphat);
    %     toc
    % 
    %     %% Compute W (P in Nature paper)(input: W, C)
    %     [P{i},W{i+1}] = blockweightvec(W{i},C{i+1},Cloc);
    % 
    %     %% Compute G and modified A (input: Img, W, mu)
    %     mu=1;
    %     tic
    %     [G{i},modW{i+1}]=modifyWvec(x,P,C{i+1},W{i+1},mu);
    %     disp(['Coarsening level ', num2str(i), ' completed'])
    %     toc
    % end


    %pause
    %%
    % Plot the cluster decisions over time.
    %
    % figure(5)
    % tiledlayout(2,1)
    % 
    % nexttile
    % plot(t,audioIn)
    % axis tight
    % ylabel("Amplitude")
    % xlabel("Time (s)")
    % 
    % nexttile
    % plot(T)
    % axis tight
    % ylabel("Cluster Index")
    % xlabel("Segment")


    %%
    % To isolate segments of speech corresponding to clusters, map the segments
    % back to audio samples. Plot the results.

    % This displays each individual method for ease of understanding.
    
    %% Print the method being used
    %fprintf('Method being used: %s\n', name)

    mask = zeros(size(audioIn,1),1);
    start = round((segmentDur/2)*fs);

    segmentHopSamples = round(segmentHopDur*fs);

    mask(1:start) = T(1);
    start = start + 1;
    for ii = 1:numel(T)
        finish = start + segmentHopSamples;
        mask(start:start + segmentHopSamples) = T(ii);
        start = finish + 1;
    end
    mask(finish:end) = T(end);

    % figure(6)
    % tiledlayout(2,1)
    % 
    % nexttile
    % plot(t,audioIn)
    % axis tight
    % 
    % nexttile
    % plot(t,mask)
    % ylabel("Cluster Index")
    % axis tight
    % xlabel("Time (s)")
    %%
    % Use <docid:audio_ref#mw_f7b40697-af02-4c71-a508-ecd8f7f47400 |detectSpeech|>
    % to determine speech regions. Use <docid:signal_ref#mw_e138186b-bef8-44a1-815d-5e067a844db4
    % |sigroi2binmask|> to convert speech regions to a binary voice activity detection
    % (VAD) mask. Call |detectSpeech| a second time without any arguments to plot
    % the detected speech regions.

    mergeDuration = 0.5;
    VADidx = detectSpeech(audioIn,fs,MergeDistance=fs*mergeDuration);

    VADmask = sigroi2binmask(VADidx,numel(audioIn));

    % figure(7)
    % detectSpeech(audioIn,fs,MergeDistance=fs*mergeDuration)
    %%
    % Apply the VAD mask to the speaker mask and plot the results. A cluster index
    % of 0 indicates a region of no speech.

    mask = mask.*VADmask;

    % figure(8)
    % tiledlayout(2,1)
    % 
    % nexttile
    % plot(t,audioIn)
    % axis tight
    % 
    % nexttile
    % plot(t,mask)
    % ylabel("Cluster Index")
    % axis tight
    % xlabel("Time (s)")

    %%
    % In this example, you assume each detected speech region belongs to a single
    % speaker. If more than two labels are present in a speech region, merge them
    % to the most frequently occuring label.

    maskLabels = zeros(size(VADidx,1),1);
    for ii = 1:size(VADidx,1)
        maskLabels(ii) = mode(mask(VADidx(ii,1):VADidx(ii,2)),"all");
        mask(VADidx(ii,1):VADidx(ii,2)) = maskLabels(ii);
    end

    % figure(9)
    % tiledlayout(2,1)
    % 
    % nexttile
    % plot(t,audioIn)
    % axis tight
    % 
    % nexttile
    % plot(t,mask)
    % ylabel("Cluster Index")
    % axis tight
    % xlabel("Time (s)")
    %%
    % Count the number of remaining speaker clusters.

    uniqueSpeakerClusters = unique(maskLabels);
    numSpeakers(i-1) = numel(uniqueSpeakerClusters);
    %% Visualize Diarization Results
    % Create a <docid:signal_ref#mw_39c843cf-ed55-4427-9070-0d55d4961689 |signalMask|>
    % object and then plot the speaker clusters. Label the plot with the ground truth
    % labels. The cluster labels are color coded with a key on the right of the plot.
    % The true labels are printed above the plot.
    



    msk = signalMask(table(VADidx,categorical(maskLabels)));

    % figure()
    % set(gcf, 'Position',  [100, 100, 1500, 200])
    % plotsigroi(msk,audioIn,true)
    % axis([0 numel(audioIn) -1 1])
    % title(name)
    %truesize([100 1000])
    
    
    %% For ICSI dataset
    trueLabel0 = categorical(participant);
    
    %% Matlab Example
    %trueLabel = groundTruth.Label;

    %%
    for ii = 1:size(VADidx,1)
        trueLabel(ii)=trueLabel0(ii);
        text(VADidx(ii,1),1.1,trueLabel(ii),FontWeight="bold")
    end
    %%
    % Choose a cluster to inspect and then use <docid:signal_ref#mw_0afffd67-38d1-434f-9067-d5ca0ec5115f
    % |binmask|> to isolate the speaker. Plot the isolated speech signal and listen
    % to the speaker cluster.

    speakerToInspect = 2;
    cutOutSilenceFromAudio = true;

    bmsk = binmask(msk,numel(audioIn));
    
    audioToPlay = audioIn;
    % if cutOutSilenceFromAudio
    %     audioToPlay(~bmsk(:,speakerToInspect)) = [];
    % end

    %sound(audioToPlay,fs)

    % figure(11)
    % tiledlayout(2,1)
    % 
    % nexttile
    % plot(t,audioIn)
    % axis tight
    % ylabel("Amplitude")
    % 
    % nexttile
    % plot(t,audioIn.*bmsk(:,speakerToInspect))
    % axis tight
    % xlabel("Time (s)")
    % ylabel("Amplitude")
    % title("Speaker Group "+speakerToInspect)

    %% Diarization System Evaluation
    % The common metric for speaker diarization systems is the diarization error
    % rate (DER). The DER is the sum of the miss rate (classifying speech as non-speech),
    % the false alarm rate (classifying non-speech as speech) and the speaker error
    % rate (confusing one speaker's speech for another).
    %
    % In this simple example, the miss rate and false alarm rate are trivial problems.
    % You evaluate the speaker error rate only.
    %
    % Map each true speaker to the corresponding best-fitting speaker cluster. To
    % determine the speaker error rate, count the number of mismatches between the
    % true speakers and the best-fitting speaker clusters, and then divide by the
    % number of true speaker regions.

    uniqueLabels = unique(trueLabel);
    guessLabels = maskLabels;
    uniqueGuessLabels = unique(guessLabels);

    totalNumErrors(i-1) = 0;
    for ii = 1:numel(uniqueLabels)
        isSpeaker = uniqueLabels(ii)==trueLabel';
        minNumErrors = inf;

        for jj = 1:numel(uniqueGuessLabels)
            groupCandidate = uniqueGuessLabels(jj) == guessLabels;
            isSpeaker;
            groupCandidate;
            if size(isSpeaker,1)>=size(groupCandidate,1)
                numErrors = nnz(isSpeaker(1:size(groupCandidate,1)) - groupCandidate);
            else 
                numErrors = nnz(isSpeaker - groupCandidate(1:size(isSpeaker,1)));
            end

            if numErrors < minNumErrors
                minNumErrors = numErrors;
                bestCandidate = jj;
            end
            minNumErrors = min(minNumErrors,numErrors);
        end
        uniqueGuessLabels(bestCandidate) = [];
        totalNumErrors(i-1) = totalNumErrors(i-1) + minNumErrors;
        if isempty(uniqueGuessLabels)
            break
        end
    end
    totalNumErrors(i-1);
    SpeakerErrorRate(i-1,n) = totalNumErrors(i-1)/numel(trueLabel);
    ErrorPercent(i-1) = totalNumErrors(i-1)/(size(trueLabel,2)*numel(trueLabel));
    
end

%TT = table(numSpeakers',totalNumErrors',SpeakerErrorRate(:,n),ErrorPercent', CPUtime(:,n));
%TT.Properties.VariableNames = ["# Speakers","Total Errors", "Error Rate","Error %", "CPU time"]
%writetable(TT,'TestRunSC')

disp(['iteration ',num2str(n), ' took ', num2str(toc(nlooptime))])
disp(['total running time so far: ', num2str(toc(totaltime))])
end
save(filename)
nlen = linspace(1,size(CPUtime,2),size(CPUtime,2));
figure
plot(nlen,CPUtime)
legend('CSS','kmeans','SC','PCA+kmeans')
title('CPU time')

nlen = linspace(1,size(SpeakerErrorRate,2),size(SpeakerErrorRate,2));
figure
plot(nlen,SpeakerErrorRate)
legend('CSS','kmeans','SC','PCA+kmeans')
title('Error Rate')

%% References
% [1] Snyder, David, et al. “X-Vectors: Robust DNN Embeddings for Speaker Recognition.”
% _2018 IEEE International Conference on Acoustics, Speech and Signal Processing
% (ICASSP)_, IEEE, 2018, pp. 5329–33. _DOI.org (Crossref)_, doi:10.1109/ICASSP.2018.8461375.
%
% [2] Sell, G., Snyder, D., McCree, A., Garcia-Romero, D., Villalba, J., Maciejewski,
% M., Manohar, V., Dehak, N., Povey, D., Watanabe, S., Khudanpur, S. (2018) Diarization
% is Hard: Some Experiences and Lessons Learned for the JHU Team in the Inaugural
% DIHARD Challenge. Proc. Interspeech 2018, 2808-2812, DOI: 10.21437/Interspeech.2018-1893.
%
%
%
% _Copyright 2020 The MathWorks, Inc._
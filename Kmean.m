addpath(genpath('.'));

%% Loading step
% loading of the files, mfeat-pix contains 240*2000
load mfeat-pix.txt -ascii;
%% initialization and assignment step
K = 200;
row = 200;
col = 240;
xtrain = mfeat_pix(1:row, 1:col);
random = randperm(200, K)
% step 1: Randomly assign k clusters.. u1, u2 ... uk E Rn
centroids = zeros(K, col);
for i = 1:K
    centroid(i, :) = xtrain(random(i), :);
end


%% repeat step


max_iter = 100;
thresh = 1e-3;

%ensures it runs the first time
delta_mu = thresh + 1;
num_iter = 0;

while(delta_mu > thresh && num_iter < max_iter)
	old_centroid = centroid;
	num_iter = num_iter + 1;
	% cluster assignment step
    cluster = [];
	for i = 1 : row
		close = [];
		for j = 1 : K
			minimum = norm(xtrain(i, :) - old_centroid(j, :));
			close = [close; minimum];
        end
		[minval, index] = min(close);
        cluster = [cluster; index];
    end     
	% move centroid step
	count = zeros(K, 1);
	add = zeros(K, col);
	for i = 1 : row
        count(cluster(i)) = count(cluster(i)) + 1;
        add(cluster(i), :) = add(cluster(i), :) + xtrain(i, :);
    end
    centroid = [];
    if count ~= 0 
        for i = 1: K
            centroid(i, :) = add(i, :) ./ count(i);
        end
	end
 
	
	delta_mu = norm(old_centroid - centroid,2);
    old_centroid = [];
end

%%

size(centroid)
% plot the figure  
figure(1);

for j = 1:min(K, 5)
    pic = centroid(j ,:);  
    picmatreverse = zeros(15,16);
    % the filling of (:) is done columnwise!
    picmatreverse(:)= - pic;
    picmat = zeros(15,16);
    for k = 1:15
        picmat(:,k)=picmatreverse(:,16-k);
    end
    %subplot(min(K, 10),10,(i-1)* 10 + count);
    subplot(1, min(K, 5), j);
    pcolor(picmat');
    axis off;
    colormap(gray(10));
end



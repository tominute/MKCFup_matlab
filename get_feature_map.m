function [out]  = get_feature_map(im_patch, features, w2c)

% out = get_feature_map(im_patch, features, w2c)
%
% Extracts the given features from the image patch. w2c is the
% Color Names matrix, if used.
cellSize=4;
hogOrientations=9;
if nargin < 2
    w2c = [];
end
% the names of the features that can be used
valid_features = {'hog', 'cn10'};

% the dimension of the valid features
feature_levels = [(cellSize-1)*hogOrientations+5, 10]';

num_valid_features = length(valid_features);
used_features = false(num_valid_features, 1);

% get the used features
for i = 1:num_valid_features
    used_features(i) = any(strcmpi(valid_features{i}, features));
end

% total number of used feature levels
num_feature_levels = sum(feature_levels .* used_features);

level = 0;

    % Features that are available for color sequances
if size(im_patch, 3) == 3 ||  used_features(1)
    % allocate space (for speed)
    out = zeros(size(im_patch, 1), size(im_patch, 2), num_feature_levels, 'single');
end
    
    % hog
if used_features(1)
    if (size(im_patch, 3) == 3)
        im_patch=rgb2gray(im_patch);
    end
    out = fhog(single(im_patch), cellSize, hogOrientations);
    out(:,:,32) =cell_grayscale(im_patch,4);

%     out = imResample(out, imgSize(1:2));
   
%     out = reshape(out, [size(out, 1)*size(out, 2), size(out, 3)]);
end
    
    % Color Names
if used_features(2)
    if size(im_patch, 3) == 1
        out = single(im_patch)/255 - 0.5;
        
    else
        if isempty(w2c)
        % load the RGB to color name matrix if not in input
            temp = load('w2crs');
            w2c = temp.w2crs;
        end
        
        % extract color descriptor
        out(:,:,level+(1:10)) = im2c(single(im_patch), w2c, -2);
%       out = bsxfun(@times, out, ...
%         sqrt((size(out,1)*size(out,2))* size(out,3) ./ ...
%         (sum(reshape(out, [], 1, 1).^2, 1) + eps))); 
%         a = sum(reshape(out, [], 1, 1).^2, 1) + eps
    end
end
end
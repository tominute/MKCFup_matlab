function [out_cn2, out_hog2] = get_subwindow(im, pos, model_sz, cn_features, hog_features ,w2c, currentScaleFactor)

% [out_npca, out_pca] = get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c)
%
% Extracts the non-PCA and PCA features from image im at position pos and
% window size sz. The features are given in non_pca_features and
% pca_features. out_npca is the window of non-PCA features and out_pca is
% the PCA-features reshaped to [prod(sz) num_pca_feature_dim]. w2c is the
% Color Names matrix if used.

if isscalar(model_sz),  %square sub-window
    model_sz = [model_sz, model_sz];
    size(model_sz)
end
patch_sz = floor(model_sz * currentScaleFactor);
if patch_sz(1) < 1
    patch_sz(1) = 2;
end;
if patch_sz(2) < 1
    patch_sz(2) = 2;
end;
xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
im_patch = im(ys, xs, :);
im_patch = mexResize(im_patch, model_sz, 'auto');
%  im_patch = imResample(im_patch, model_sz(1:2));
% compute hog feature map
if ~isempty(hog_features)
    [out_hog] = get_feature_map(im_patch, hog_features, w2c);
  
%    out_hog = imResample(out_hog, model_sz(1:2));
    out_hog2 = reshape(out_hog, [size(out_hog, 1)*size(out_hog, 2), size(out_hog, 3)]);
else
    out_hog2 = [];
end

% compute cn feature map
if ~isempty(cn_features)
    [out_cn] = get_feature_map(im_patch, cn_features, w2c);
    out_cn = average_feature_region(out_cn, 4);
    out_cn2 = reshape(out_cn, [size(out_cn, 1)*size(out_cn, 2), size(out_cn, 3)]);
else
    out_cn2 = [];
end
% compute gray feature map

end


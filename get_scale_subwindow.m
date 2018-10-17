function [out_pca,out_npca] = get_scale_subwindow(im, pos, base_target_sz, scaleFactors, scale_model_sz, w2c)

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
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
    
    % resize image to model size
%     im_patch_resized = imresize(im_patch, scale_model_sz, 'bilinear');
%     im_patch_resized = imResample(im_patch, scale_model_sz(1:2));
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
%     temp_hog = imResample(temp_hog, scale_model_sz(1:2));
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if size(im_patch_resized, 3) == 1
%         temp_cn = single(im_patch_resized)/255 - 0.5;
%         
%     else
%         if isempty(w2c)
%         % load the RGB to color name matrix if not in input
%             temp = load('w2crs');
%             w2c = temp.w2crs;
%         end
%         
%         % extract color descriptor
%         temp_cn(:,:,(1:10)) = im2c(single(im_patch_resized), w2c, -2);
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      temp_fea = cat(3, temp_hog, temp_cn);
     if s == 1
        dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;
        out_pca = zeros(dim_scale, nScales, 'single');
    end
    out_pca(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
 
end

out_npca = [];
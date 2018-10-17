function [z_cn,z_hog] = feature_projection(x_cn,x_hog, projection_matrix_cn,projection_matrix_hog, cos_window)

% z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)
%
% Calculates the compressed feature map by mapping the PCA features with
% the projection matrix and concatinates this with the non-PCA features.
% The feature map is then windowed.

% get dimensions
[height, width] = size(cos_window );
if size(x_cn,2) == 10
    [~, num_pca_out_cn] = size(projection_matrix_cn);
    x_proj_cn = reshape(x_cn * projection_matrix_cn, [height, width, num_pca_out_cn]);
else
    x_proj_cn = reshape(x_cn, [height, width, 1]);
end

[~, num_pca_out_hog] = size(projection_matrix_hog);
x_proj_hog = reshape(x_hog * projection_matrix_hog, [height, width, num_pca_out_hog]);
    
% project the PCA-features using the projection matrix and reshape
% to a window

% concatinate the feature windows
    
z_cn = x_proj_cn;
z_hog = x_proj_hog;
z_hog = bsxfun(@times, cos_window, z_hog);

% do the windowing of the output
z_cn = bsxfun(@times, cos_window, z_cn);
end
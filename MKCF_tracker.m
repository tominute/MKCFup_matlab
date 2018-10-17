function [positions, fps, tran] = MKCF_tracker(params)
close all;
% [positions, fps] = color_tracker(params)

% parameters
padding = params.padding;
output_sigma_factor = params.output_sigma_factor;
% sigma = params.sigma;
 lambda = params.lambda;
learning_rate_cn_color = params.learning_rate_cn_color;
learning_rate_cn_gray = params.learning_rate_cn_gray;
learning_rate_hog_color = params.learning_rate_hog_color;
learning_rate_hog_gray = params.learning_rate_hog_gray;
debug = params.debug;
%compression_learning_rate = params.compression_learning_rate;
cn_features = params.cn_features;
hog_features = params.hog_features;
num_compressed_dim_cn = params.num_compressed_dim_cn;
num_compressed_dim_hog = params.num_compressed_dim_hog;
interp_factor = params.interp_factor;
refinement_iterations = params.refinement_iterations;
translation_model_max_area = params.translation_model_max_area;
nScales = params.number_of_scales;
nScalesInterp = params.number_of_interp_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_factor = params.scale_model_factor;%1
scale_model_max_area = params.scale_model_max_area;
interpolate_response = params.interpolate_response;
cnSigma_color = params.cnSigma_color;
hogSigma_color = params.hogSigma_color;
cnSigma_gray = params.cnSigma_gray;
hogSigma_gray = params.hogSigma_gray;

video_path = params.video_path;
img_files = params.img_files;
pos = floor(params.init_pos);
old_pos = pos;
target_sz = floor(params.wsize);
init_target_sz = target_sz;
visualization = params.visualization;

num_frames = numel(img_files);
if prod(init_target_sz) > translation_model_max_area
    currentScaleFactor = sqrt(prod(init_target_sz) / translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% load the normalized Color Name matrix
temp = load('w2crs');
w2c = temp.w2crs;
old_ScaleFactor =1;
% window size, taking padding into account
sz = floor( base_target_sz * (1 + padding ));
% if sz(1)-base_target_sz(1) <= sz(2)-base_target_sz(2)
%     sz(2) = base_target_sz(2)+sz(1)-base_target_sz(1);
% else
%     sz(1) = base_target_sz(1)+sz(2)-base_target_sz(2);
% end
% sz = zeros(1,2);
% sz(1) = floor(sqrt(prod(base_target_sz)*(1.5 + padding )*(1.5 + padding )));
% if sz(1) < max(base_target_sz)+20
%     sz(1) = max(base_target_sz)+20;
% end
% sz(2) = sz(1);
% desired output (gaussian shaped), bandwidth proportional to target size
featureRatio = 4;

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;

use_sz = floor(sz/featureRatio);
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);

[rs, cs] = ndgrid( rg,cg);
y = 0.5*exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

interp_sz = size(y) * featureRatio;
% store pre-computed cosine window
cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );
% cos_window = single(kaiser(floor(sz(1)/featureRatio),4)*kaiser(floor(sz(2)/featureRatio),4)' );
% imshow(cos_window)
im = imread([video_path img_files{1}]);
if nScales > 0 %17
    scale_sigma = nScalesInterp * scale_sigma_factor;%33*1/16
    
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);
    
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
    
    scaleSizeFactors = scale_step .^ scale_exp;%1.02
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;
    
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
    ysf = single(fft(ys));
   
    scale_window = single(hann(size(ysf,2)))';
    %make sure the scale model is not to large, to save computation time
    if scale_model_factor^2 * prod(init_target_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
    end
    
    %set the scale model size
    scale_model_sz = floor(init_target_sz * scale_model_factor);
    
  
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
    
    max_scale_dim = strcmp(params.s_num_compressed_dim,'MAX');
    if max_scale_dim
        s_num_compressed_dim = length(scaleSizeFactors);
    else
        s_num_compressed_dim = params.s_num_compressed_dim;
    end
end
if size(im,3) == 3
    cnSigma = cnSigma_color;
    hogSigma = hogSigma_color;
    learning_rate_hog = learning_rate_hog_color;
    learning_rate_cn = learning_rate_cn_color;
    modnum = params.gap;
else
    cnSigma = cnSigma_gray;
    hogSigma = hogSigma_gray;
    learning_rate_hog = learning_rate_hog_gray;
    learning_rate_cn = learning_rate_cn_gray;
    modnum = 1;
end
% to calculate precision
positions = zeros(numel(num_frames), 4);
temp_res_psr = 20;
% initialize the projection matrix
projection_matrix_cn2 = [];
projection_matrix_cn = [];
projection_matrix_hog = [];
rect_position = zeros(num_frames, 4);
res_psr = 20;
% to calculate fps
time = 0;
roww = zeros(1,5);
coll = zeros(1,5);
maxx = zeros(1,5);
res_psrr = zeros(1,5);
num_den = 8;
% a=1;
tran = zeros(num_frames); 

for frame = 1:num_frames
% for frame = 1:2
% %     load image
%     if frame ==1
        im = imread([video_path img_files{frame}]);
%     else
%         init = im; % 读取图像
%         [R, C] = size(init); % 获取图像大小
%         res = zeros(R, C); % 构造结果矩阵。每个像素点默认初始化为0（黑色）
%         delX = round(sz(1)*0.25); % 平移量X
%         delY = 0; % 平移量Y
%         tras = [1 0 delX; 0 1 delY; 0 0 1]; % 平移的变换矩阵
%          
%             for i = 1 : R
%                 for j = 1 : C
%                     temp = [i; j; 1];
%                     temp = tras * temp; % 矩阵乘法
%                     x = temp(1, 1);
%                     y = temp(2, 1);
%                     % 变换后的位置判断是否越界
%                     if (x <= R) && (y <= C) && (x >= 1) && (y >= 1)
%                         res(x, y) = init(i, j);
%                     end
%                 end
%             end
%        
%         im = uint8(res);
%     end
%     imshow(im)
    tic;
    
    if frame > 1
        % compute the compressed learnt appearance
        old_pos = inf(size(pos));
        iter = 1;
        %translation search
        
        while iter <= refinement_iterations && any(old_pos ~= pos)
            %[zp,z_hog] = feature_projection(z_cn, z_gray,z_hog, projection_matrix, cos_window);
%             if prod(sz*currentScaleFactor) > 0.5*size(im,1)*size(im,2)
%                 a = 0.8;
%             end
%             if res_psr< 10
%                 scale_pos = 1.2;
%             else
%                 scale_pos = 1;
%             end
            % extract the feature map of the local image patch
%             trans = [0,0;floor(sz(1)*currentScaleFactor/num_den),floor(sz(2)*currentScaleFactor/num_den);-floor(sz(1)*currentScaleFactor/num_den) ,...
%                 floor(sz(2)*currentScaleFactor/num_den);floor(sz(1)*currentScaleFactor/num_den),...
%                 -floor(sz(2)*currentScaleFactor/num_den);-floor(sz(1)*currentScaleFactor/num_den), -floor(sz(2)*currentScaleFactor/num_den)];
        
           
            [xo_cn, xo_hog] = get_subwindow(im, pos, sz, cn_features, hog_features, w2c, currentScaleFactor);

            % do the dimensionality reduction and windowing
            [xo_cn2,xo_hog2] = feature_projection(xo_cn, xo_hog, projection_matrix_cn,projection_matrix_hog, cos_window);
            % calculate the response of the classifier
            detect_k_cn = (dense_gauss_kernel(cnSigma, z_cn2, xo_cn2));
            detect_k_hog = (dense_gauss_kernel(hogSigma, z_hog2, xo_hog2));
            kf=fft2(d(1)*detect_k_cn +d(2)*detect_k_hog);
            responsef = alphaf.*conj(kf);

            if interpolate_response > 0
                if interpolate_response == 2
                    % use dynamic interp size
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end   
                responsef = resizeDFT2(responsef, interp_sz);
            end                
            response = ifft2(responsef, 'symmetric');           
%             res_psr = calcPsr(response);

%             if frame > 30
%                 detect_k_cn = (dense_gauss_kernel(cnSigma, temp_z_cn2, xo_cn2));
%                 detect_k_hog = (dense_gauss_kernel(hogSigma, temp_z_hog2, xo_hog2));
%                 kf=fft2(temp_d(1)*detect_k_cn +temp_d(2)*detect_k_hog);
%                 temp_responsef = temp_alphaf.*conj(kf);
% 
%                 if interpolate_response > 0
%                     if interpolate_response == 2
%                         % use dynamic interp size
%                         interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
%                     end
%                     temp_responsef = resizeDFT2(temp_responsef, interp_sz);
%                 end
%                 temp_response = ifft2(temp_responsef, 'symmetric');
%                 temp_res_psr = calcPsr(temp_response);
%             end

            if debug
                figure(2);
                imagesc(fftshift(response(:,:)));colorbar; axis image;
                title(sprintf('max(response) = %f,var=%f', max(max(response(:))),res_psr));
            end            
            % target location is at the maximum response
            [row, col] = find(response == max(response(:)), 1);
  
%             if max(response(:)) < 0.3 && frame>30
%                 for i =1:4
%                     [xo_cn, xo_hog] = get_subwindow(im, pos+trans(i,:), sz, cn_features, hog_features, w2c,currentScaleFactor);
%                     
%                     % do the dimensionality reduction and windowing
%                     [xo_cn2,xo_hog2] = feature_projection(xo_cn, xo_hog, projection_matrix_cn,projection_matrix_hog, cos_window );
%                     
%                     % calculate the response of the classifier
%                     detect_k_cn = (dense_gauss_kernel(cnSigma, temp_z_cn2, xo_cn2));
%                     detect_k_hog = (dense_gauss_kernel(hogSigma, temp_z_hog2, xo_hog2));
%                     kf=fft2(temp_d(1)*detect_k_cn +temp_d(2)*detect_k_hog);
%                     responsef = temp_alphaf.*conj(kf);
%                     if interpolate_response > 0
%                         if interpolate_response == 2
%                             % use dynamic interp size
%                             interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
%                         end
%                         
%                         responsef = resizeDFT2(responsef, interp_sz);
%                     end
%                     
%                     response = ifft2(responsef, 'symmetric');
%                     res_psrr(i) = calcPsr(response);
%                     maxx(i) = max(response(:));
%                     [roww(i), coll(i)] = find(response == max(response(:)), 1);
%                 end
%                 index = find(maxx == max(maxx(:)), 1);
%                 row = roww(index);
%                 col = coll(index);
%                 pos = pos+trans(index,:);
%                 res_psr = res_psrr(index);
%               
%             end
            
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            switch interpolate_response
                case 0
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
                case 1
                    translation_vec = round([disp_row, disp_col] *currentScaleFactor);
                case 2
                    translation_vec = [disp_row, disp_col];
            end
            trans = sqrt(sz(1)*sz(2))*currentScaleFactor/3;
            
            old_pos = pos;
            pos = pos + translation_vec;
            
            iter = iter + 1;
        end
%         if res_psr < 10
%             pos = old_pos;
%         end
        
        if nScales > 0 
            
              %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, w2c);
           
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
            xsf = fft(xs,[],2);

            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);
         
            interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
            
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
            old_ScaleFactor = currentScaleFactor;
            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end
    end
    %%start comprehension
    % extract the feature map of the local image patch to train the classifer
%     if frame==1
%         pos=pos+[1 1];
%     end
%     if frame == 1 || mod(frame,) == 0
        [xo_cn, xo_hog] = get_subwindow(im, pos, sz, cn_features, hog_features, w2c,currentScaleFactor);
%         if res_psr <= 15 && count == 0
%             old_frame = frame;
%             count = count + 1;
%         end
%         if res_psr <= 15 && (old_frame - frame) == 1 && count ~= 0
%             old_frame = frame;
%             count = count + 1;
%         end
      
        if res_psr <= 10 
%             count = 0;
            learning_rate = 1;
        else
            learning_rate = 1;
        end
        if frame == 1
            % initialize the appearance
            z_hog = xo_hog;
            z_cn = xo_cn;
        else
            % update the appearance
            if size(im,3)==3
                z_hog = (1 - learning_rate_hog*learning_rate) * z_hog + learning_rate_hog*learning_rate * xo_hog;
            else
                z_hog = (1 - learning_rate_hog*learning_rate) * z_hog + learning_rate_hog*learning_rate * xo_hog;
            end
            z_cn = (1 - learning_rate_cn*learning_rate) * z_cn + learning_rate_cn*learning_rate * xo_cn;
        end

        % if dimensionality reduction is used: update the projection matrix
        if size(im,3) == 3
            data_matrix_cn = z_cn;
            [pca_basis_cn, ~, ~] = svd(data_matrix_cn' * data_matrix_cn);
            projection_matrix_cn = pca_basis_cn(:, 1:num_compressed_dim_cn);
        end
        data_matrix_hog = z_hog;
        [pca_basis_hog, ~, ~] = svd(data_matrix_hog' * data_matrix_hog);
        projection_matrix_hog = pca_basis_hog(:, 1:num_compressed_dim_hog);
        % project the features of the new appearance example using the new
        % projection matrix
        [z_cn2,z_hog2] = feature_projection(z_cn, z_hog, projection_matrix_cn, projection_matrix_hog, cos_window);
        
        % calculate the new classifier coefficients
        if frame ==1
            alphaf_num1=[];
            alphaf_num2=[];
            alphaf_den1=[];
            alphaf_den2=[];
            d_num1=[];
            d_num2=[];
            d_den1=[];
            d_den2=[];
        [alphaf,d,alphaf_num1,alphaf_num2,alphaf_den1,alphaf_den2,d_num1,d_num2,d_den1,d_den2]=trainModel(z_cn2,z_hog2,yf,frame,alphaf_num1,...
            alphaf_num2,alphaf_den1,alphaf_den2,learning_rate_hog,learning_rate_cn ,d_num1,d_num2,d_den1,d_den2,cnSigma,hogSigma);
        
        elseif mod(frame,modnum) == 0
            
            [alphaf,d,alphaf_num1,alphaf_num2,alphaf_den1,alphaf_den2,d_num1,d_num2,d_den1,d_den2]=trainModel(z_cn2,z_hog2,yf,frame,alphaf_num1,...
                alphaf_num2,alphaf_den1,alphaf_den2,learning_rate_hog,learning_rate_cn ,d_num1,d_num2,d_den1,d_den2,cnSigma,hogSigma);
        
        end
        if nScales > 0

            %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, w2c);

            if frame == 1
                s_num = xs_pca;
            else
                s_num = (1 - interp_factor) * s_num + interp_factor * xs_pca;
            end;

            bigY = s_num;
            bigY_den = xs_pca;
   
            if max_scale_dim
                [scale_basis, ~] = qr(bigY, 0); 
                [scale_basis_den, ~] = qr(bigY_den, 0);
            else
                [U,~,~] = svd(bigY,'econ');             
                [Ud,~,~] = svd(bigY_den,'econ');
                scale_basis = U(:,1:s_num_compressed_dim);
                scale_basis_den = Ud(:,1:s_num_compressed_dim);
            end
            scale_basis = scale_basis';
      
            %create the filter update coefficients
            sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
   
            sf_num = bsxfun(@times,ysf,conj(sf_proj));

            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
            xsf = fft(xs,[],2);
            new_sf_den = sum(xsf .* conj(xsf),1);

            if frame == 1
                sf_den = new_sf_den;
            else
                sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den;
            end;
        end
        
%     end
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(frame,:) = [pos([1,2]) , target_sz([1,2])];
    if frame > 1
        tran(frame) = trans;
    end
    time = time + toc;
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        rect_position_pad = [old_pos([2,1]) - sz([2,1])*old_ScaleFactor/2, sz([2,1])*old_ScaleFactor];
        if frame == 1
            figure( 'NumberTitle','off','Name',['Tracker - ' video_path]);
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g');
            rect_handle_pad = rectangle('Position',rect_position_pad, 'EdgeColor','r');
            text_handle = text(10, 10, [int2str(frame) '/' int2str(num_frames)]);
            set(text_handle, 'color', [0 1 1]);
           
                text_handle2 = text(10, 40, int2str(temp_res_psr));
                set(text_handle2, 'color', [0 1 1]);

        else
            try
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position_vis)
                set(rect_handle_pad, 'Position', rect_position_pad)
                set(text_handle, 'string', [int2str(frame) '/' int2str(num_frames)]);
           
                    set(text_handle2, 'string', int2str(temp_res_psr));

            catch
                return
            end
        end
        
        drawnow
        %pause
    end
end
positions=rect_position;
fps = num_frames/time;
tran;
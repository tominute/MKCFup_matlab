
% run_tracker.m

close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'F:/Project/OTB100/sequences/';

OTB2013Set = {'Basketball', 'Doll', 'Bolt', 'Soccer', 'Deer', 'Boy', 'CarDark', 'CarScale',...
          'David', 'David3', 'Football1', 'Girl', 'Crossing',...
          'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1',...
          'Trellis', 'Walking', 'Walking2', 'Woman', 'Tiger1', 'MotorRolling',...
          'Lemming', 'Matrix', 'Coke', 'FaceOcc1', 'Liquor', 'Skiing', 'Tiger2', 'Ironman', 'Couple',...
          'Car4', 'Subway', 'David2', 'Dog1', 'Dudek', 'FaceOcc2', 'Fish', 'FleetFace',...
          'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Jogging-1', 'Jogging-2',...
          'Jumping', 'Mhyang', 'Suv', 'Sylvester'};
OTB100Set = {'Basketball', 'Doll', 'Bolt', 'Soccer', 'Deer', 'Boy', 'CarDark', 'CarScale',...
          'David', 'David3', 'Football1', 'Girl', 'Crossing',...
          'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1',...
          'Trellis', 'Walking', 'Walking2', 'Woman', 'Tiger1', 'MotorRolling',...
          'Lemming', 'Matrix', 'Coke', 'FaceOcc1', 'Liquor', 'Skiing', 'Tiger2', 'Ironman', 'Couple',...
          'Car4', 'Subway', 'David2', 'Dog1', 'Dudek', 'FaceOcc2', 'Fish', 'FleetFace',...
          'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Jogging-1', 'Jogging-2',...
          'Jumping', 'Mhyang', 'Suv', 'Sylvester','Biker','Bird1','Bird2','BlurBody','BlurCar1','BlurCar2','BlurCar3','BlurCar4',...
          'BlurFace','BlurOwl','Board','Bolt2','Box','Car1','Car2','Car24','ClifBar','Coupon','Crowds','Dancer','Dancer2',...
          'Diving','Dog','DragonBaby','Girl2','Gym','Human2','Human3','Human4-2','Human5','Human6','Human7','Human8','Human9',...
          'Jump','KiteSurf', 'Man','Panda','RedTeam','Rubik','Skater','Skater2','Skating2-1','Skating2-2','Surfer','Toy','Trans','Twinnings','Vase'};

testSet = OTB2013Set;  %OTB100rmSet
user_choose = 1;
params.debug = 0;

params.gap = 6;

params.learning_rate_cn_color = 0.0174;
params.learning_rate_cn_gray = 0.0175;
params.learning_rate_hog_color = 0.0173;%0.0175(for OTB100)
params.learning_rate_hog_gray = 0.018;
params.num_compressed_dim_cn = 4;
params.num_compressed_dim_hog = 4;
%parameters according to the paper
params.padding = 1.5;                       % extra area surrounding the target
params.output_sigma_factor = 1/16;          % spatial bandwidth (proportional to target)
params.scale_sigma_factor = 1/16;           % standard deviation for the desired scale filter output
params.lambda = 1e-2;                       %  Scale regularization (denoted "lambda" in the paper)
params.interp_factor = 0.025;               % tracking model learning rate (denoted "eta" in the paper)
params.cn_features = {'cn10'};              % features that atranslation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);re not compressed, a cell with strings (possible choices: 'gray', 'cn')
params.hog_features = {'hog'};              % features that are compressed, a cell with strings (possible choices: 'gray', 'cn')

params.cnSigma_color = 0.515;
params.hogSigma_color = 0.6;

params.cnSigma_gray = 0.3;
params.hogSigma_gray = 0.4;
params.refinement_iterations = 1;           % number of iterations used to refine the resulting position in a frame
params.translation_model_max_area = inf;    % maximum area of the translation model
params.interpolate_response = 1;            % interpolation method for the translation scores
params.lamda=0.01;
params.number_of_scales = 20;               % number of scale levels
params.number_of_interp_scales = 39;        % number of scale levels after interpolation
params.scale_model_factor = 1.0;            % relative size of the scale sample
params.scale_step = 1.02;                   % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;          % the maximum size of scale examples
params.s_num_compressed_dim = 'MAX';        % number of compressed scale feature dimensions'MAX'

params.visualization = 0;

%ask the user for the video
if user_choose==0
    video_path = choose_video(base_path);
    if isempty(video_path), return, end     %user cancelled
    [img_files, pos, target_sz, ground_truth, video_path] = ...
        load_video_info(video_path);
    params.video_path = video_path;
    params.init_pos = floor(pos) + floor(target_sz/2);
    params.wsize = floor(target_sz);
    params.img_files = img_files;
    [positions, fps] = MKCF_tracker(params);
    % calculate precisions
    [distance_precision, PASCAL_precision, average_center_location_error,Overlap, flag] = ...
        compute_performance_measures(positions, ground_truth);

    fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nOverlap: %.3g%%\nSpeed: %.3g fps\n', ...
        average_center_location_error, 100*distance_precision, 100*PASCAL_precision,100*Overlap, fps);

else
    average_center_location_error_sum=0;
    distance_precision_sum=0;
    PASCAL_precision_sum=0;
    Overlap_sum=0;
    fps_sum=0;
    for i = 1:length(testSet)
        video_path = [base_path testSet{i} '/'];
        [img_files, pos, target_sz, ground_truth, video_path] = ...
            load_video_info(video_path);
        params.video_path = video_path;
        params.init_pos = floor(pos) + floor(target_sz/2);
        params.wsize = floor(target_sz);
        params.img_files = img_files;
        [positions, fps, tran] = MKCF_tracker(params);
        [distance_precision, PASCAL_precision, average_center_location_error, Overlap, flag] = ...
            compute_performance_measures(positions, ground_truth, 20, 0.5, tran);        
        disp([testSet{i} ': ' num2str(Overlap)]);
        average_center_location_error_sum=average_center_location_error_sum+average_center_location_error;
        distance_precision_sum=distance_precision_sum+distance_precision;
        PASCAL_precision_sum=PASCAL_precision_sum+PASCAL_precision;
        Overlap_sum=Overlap_sum+Overlap;
        fps_sum=fps_sum+fps;
        
    end
    average_center_location_error=average_center_location_error_sum/length(testSet);
    distance_precision=distance_precision_sum/length(testSet);
    PASCAL_precision=PASCAL_precision_sum/length(testSet);
    Overlap=Overlap_sum/length(testSet);
    fps=fps_sum/length(testSet);
    fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.5g %%\nOverlap: %.5g%%\nSpeed: %.5g fps\n', ...
        average_center_location_error, 100*distance_precision, 100*PASCAL_precision,100*Overlap, fps);
end

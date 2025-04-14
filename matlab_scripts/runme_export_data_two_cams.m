%% User inputs
% exp_id = 23;
cam_names = {'cam_0', 'cam_1'};
mean_reprojection_error = 0.07; % Reprojection error of real cameras, to decide a threshold to identify stationary targets
frame_rate = 100;

% Choose calibration target type from
% (1) 'dotted_grid'
% (2) 'laser_spot'
% (3) 'ball_bearing'
calibration_target_type = 'dotted_grid';
% num_images_list = [3048, 2027, 1665, 1645, 1758, 2110, 2109, 1730, 2100, 3645]; % NOTE: To be deleted

triangulate_calibration_grid = false;
triangulate_laser_spot = true;
export_triangulated_points = true;
make_scatter_plot = false;
grid_size = [9, 9];
grid_spacing = 1;

%% Load data
% dataDir = []; % Get this from the bash script (path to prism_new_led/)
results_dir = [dataDir, '/exp_', num2str(exp_id), '/results/'];

% Load camera parameters
camera_parameters_path = [results_dir, 'camera_parameters_', cam_names{1} ,'_', cam_names{2}, '.mat'];
load(camera_parameters_path)
display(['Analyzing for target type : ', calibration_target_type])

% Load grid parameters  
switch calibration_target_type
    case 'dotted_grid'
        target_coordinates_path = [results_dir, '/grid_points_ref.mat'];
        load(target_coordinates_path)        
        score00 = cell(size(imagePoints_a, 2), 1); % backward compatibility of ball bearings with grid and laser
        score02 = cell(size(imagePoints_a, 2), 1); % backward compatibility of ball bearings with grid and laser
        score11 = cell(size(imagePoints_a, 2), 1); % backward compatibility of ball bearings with grid and laser
        score13 = cell(size(imagePoints_a, 2), 1); % backward compatibility of ball bearings with grid and laser
        ball_size_list = zeros(length(imagePoints_a), 1);
        for i = 1:length(score00)
            score00{i} = ones(size(imagePoints_a{i}, 1), 1);
            score11{i} = ones(size(imagePoints_b{i}, 1), 1);
            score02{i} = ones(size(imagePoints_a{i}, 1), 1);
            score13{i} = ones(size(imagePoints_b{i}, 1), 1);
            score_array00{i} = 0;
            score_array02{i} = 0;
            score_array11{i} = 0;
            score_array13{i} = 0;
        end
    case 'laser_spot'
        load(target_coordinates_path)
        target_coordinates_path = [results_dir, '/grid_parameters_', cam_names{1} ,'_', cam_names{2}, '.mat'];
        score00 = cell([size(imagePoints_a, 1), size(imagePoints_a, 2)]); % backward compatibility of ball bearings with grid and laser
        score11 = cell([size(imagePoints_a, 1), size(imagePoints_a, 2)]);
    case 'ball_bearing'
        % NOTE: This path needs to change
        ball_size_list = [10, 20, 25, 30, 35, 40, 45, 50, 55];
        for ball_id = 1:length(ball_size_list)
            ball_size = ball_size_list(ball_id);
            target_coordinates_path = [results_dir, '/results_', num2str(ball_size), '.mat'];
            load(target_coordinates_path)
            imagePoints_a{ball_id} = results.centers00;
            imagePoints_b{ball_id} = results.centers11;
            imagePoints_av{ball_id} = results.centers02;
            imagePoints_bv{ball_id} = results.centers13;
            score00{ball_id} = mat2gray(results.hough_score00);
            score02{ball_id} = mat2gray(results.hough_score02);
            score11{ball_id} = mat2gray(results.hough_score11);
            score13{ball_id} = mat2gray(results.hough_score13);
        end
        %% Threshold score array: user-defined
        if make_scatter_plot
            for i = 1:length(score00)
                figure, title(num2str(i))
                subplot(4,1,1), histogram(score00{i})
                subplot(4,1,2), histogram(score02{i})
                subplot(4,1,3), histogram(score11{i})
                subplot(4,1,4), histogram(score13{i})
            end
        end


        %%
        score_array00{1} = 0.7;
        score_array02{1} = 0.6;
        score_array11{1} = 0.7;
        score_array13{1} = 0.6;

        score_array00{2} = 0.7;
        score_array02{2} = 0.6;
        score_array11{2} = 0.7;
        score_array13{2} = 0.6;


        score_array00{3} = 0.7;
        score_array02{3} = 0.45;
        score_array11{3} = 0.8;
        score_array13{3} = 0.6;

        score_array00{4} = 0.75;
        score_array02{4} = 0.6;
        score_array11{4} = 0.7;
        score_array13{4} = 0.6;

        score_array00{5} = 0.5;
        score_array02{5} = 0.4;
        score_array11{5} = 0.6;
        score_array13{5} = 0.6;

        score_array00{6} = 0.6;
        score_array02{6} = 0.5;
        score_array11{6} = 0.6;
        score_array13{6} = 0.5;

        score_array00{7} = 0.72;
        score_array02{7} = 0.4;
        score_array11{7} = 0.6;
        score_array13{7} = 0.5;

        score_array00{8} = 0.6;
        score_array02{8} = 0.55;
        score_array11{8} = 0.6;
        score_array13{8} = 0.5;

        score_array00{9} = 0.6;
        score_array02{9} = 0.55;
        score_array11{9} = 0.7;
        score_array13{9} = 0.65;
end

%% Triangulate and (optionally - flag below) plot 3-D target points
plot_triangulated_points = false;
imagePoints_a_undistorted = cell(size(imagePoints_a));
imagePoints_av_undistorted = cell(size(imagePoints_a));
imagePoints_b_undistorted = cell(size(imagePoints_b));
imagePoints_bv_undistorted = cell(size(imagePoints_b));
worldPoints = cell(length(imagePoints_a), 1);
% reprojection_errors = zeros(size(imagePoints_a,1), size(imagePoints_a,3));
acceptable_detect_id = cell(length(ball_size_list), 1);
colors_bad_detections = 'r';
colors_acceptable_detections =  'g';
colors = jet(size(imagePoints_a, 2));

% figure,
for i = 1:size(imagePoints_b, 2)
    display(['Triangulating points from ', num2str(ball_size_list(i) / 10), 'mm ball bearing. ', num2str(i), ...
        ' of ', num2str(size(imagePoints_b, 2))])
    hough_threshold00 = graythresh(score00{i});
    hough_threshold02 = graythresh(score02{i});
    hough_threshold11 = graythresh(score11{i});
    hough_threshold13 = graythresh(score13{i});
    hough_threshold00 = score_array00{i};
    hough_threshold02 = score_array02{i};
    hough_threshold11 = score_array11{i};
    hough_threshold13 = score_array13{i};
    imagePoints_a_undistorted{i} = undistortPoints(imagePoints_a{i}, cameraParams{1});
    imagePoints_av_undistorted{i} = undistortPoints(imagePoints_av{i}, cameraParams{1});
    imagePoints_b_undistorted{i} = undistortPoints(imagePoints_b{i}, cameraParams{2});
    imagePoints_bv_undistorted{i} = undistortPoints(imagePoints_bv{i}, cameraParams{2});
    [worldPoints{i}, reprojection_errors{i}] = triangulate(imagePoints_a_undistorted{i}, imagePoints_b_undistorted{i}, stereoParams);
    
    
    % Filter bad detections from hough transform
    % Threshold hough score is based on Otsu on the histogram of scores
    bad_detect_id = find(score00{i} < hough_threshold00 & score11{i} < hough_threshold11 & ...
        score02{i} < hough_threshold02 & score13{i} < hough_threshold13); % reject points with bad target detections

    % Filter frames where the ball is not moving
    % Threshold moving distance (pixels) is decided based on the mean reprojection
    % errors
%     acceptable_detect_id{i} = intersect(find(score00{i} >= hough_threshold00 & score11{i} >= hough_threshold11 & ...
%         score02{i} >= hough_threshold02 & score13{i} >= hough_threshold13), ...
%         reject_stationary_points(worldPoints{i}, mean_reprojection_error, frame_rate));

    acceptable_detect_id{i} = (find(score00{i} >= hough_threshold00 & score11{i} >= hough_threshold11 & ...
        score02{i} >= hough_threshold02 & score13{i} >= hough_threshold13));

    if plot_triangulated_points
        scatter3(worldPoints{i}(acceptable_detect_id{i}, 1), worldPoints{i}(acceptable_detect_id{i}, 2), worldPoints{i}(acceptable_detect_id{i}, 3),...
            'Marker', '.', 'MarkerEdgeColor', colors(i,:))

        hold on
        scatter3(worldPoints{i}(bad_detect_id, 1), worldPoints{i}(bad_detect_id, 2), worldPoints{i}(bad_detect_id, 3), ...
            'Marker', '.', 'MarkerEdgeColor', colors_bad_detections)

        axis padded
        view(40, 40)
        xlabel('X (mm)')
        ylabel('Y (mm)')
        zlabel('Z (mm) - Camera axis')
        set(gca, 'FontSize', 32)
    end
    axis('equal')
end


%%

%% Triangulation plots with error-based color labels
% 
% figure, 
% colors = flip(hot(100));
% normalized_error = ceil(100* reprojection_errors / max(reprojection_errors));
% hold on
% for n = 1:size(imagePoints_a,1)
%     plot3(worldPoints(n, 1), worldPoints(n, 2), worldPoints(n, 3), '.', 'color', colors(normalized_error(n), :),...
%         'DisplayName', num2str(n))
%     hold on
% end
% colormap(flip(hot(100)))
% xlabel('X (mm)')
% ylabel('Y (mm)')
% zlabel('Z (mm) - Camera axis')
% axis('equal')
% set(gca, 'FontSize', 32)

%% Accumulate filtered frames where the balls are not moving
acceptable_reprojection_errors_all = [];
acceptable_reprojection_errors = {};
input_data_moving = [];
output_data_cam_0_moving = [];
output_data_cam_1_moving = [];
output_data_cam_02_moving = [];
output_data_cam_13_moving = [];
output_data_cam_0_moving_undistorted = [];
output_data_cam_1_moving_undistorted = [];
output_data_cam_02_moving_undistorted = [];
output_data_cam_13_moving_undistorted = [];
mkdir([results_dir, 'figures/'])
figure, 
for i = 1:size(imagePoints_a, 2)
    acceptable_reprojection_errors_all = [acceptable_reprojection_errors_all; reprojection_errors{i}(acceptable_detect_id{i})];
    acceptable_reprojection_errors{i} = reprojection_errors{i}(acceptable_detect_id{i});
    display(['Mean reprojection error for ', num2str(i), 'th ball of size ', num2str(ball_size_list(i)),...
        ' : ', num2str(mean(reprojection_errors{i}(acceptable_detect_id{i})))])
    acceptable_reprojection_errors{i} = reprojection_errors{i}(acceptable_detect_id{i});
    input_data_moving = [input_data_moving; worldPoints{i}(acceptable_detect_id{i}, :)];
    output_data_cam_0_moving = [output_data_cam_0_moving; ...
        imagePoints_a{i}(acceptable_detect_id{i},:)];
    output_data_cam_1_moving = [output_data_cam_1_moving; ...
        imagePoints_b{i}(acceptable_detect_id{i}, :)];
    output_data_cam_02_moving = [output_data_cam_02_moving; ...
        imagePoints_av{i}(acceptable_detect_id{i},:)];
    output_data_cam_13_moving = [output_data_cam_13_moving; ...
        imagePoints_bv{i}(acceptable_detect_id{i}, :)];
    output_data_cam_0_moving_undistorted = [output_data_cam_0_moving_undistorted; ...
        imagePoints_a_undistorted{i}(acceptable_detect_id{i}, :)];
    output_data_cam_1_moving_undistorted = [output_data_cam_1_moving_undistorted; ...
        imagePoints_b_undistorted{i}(acceptable_detect_id{i}, :)];
    output_data_cam_02_moving_undistorted = [output_data_cam_02_moving_undistorted; ...
        imagePoints_av_undistorted{i}(acceptable_detect_id{i}, :)];
    output_data_cam_13_moving_undistorted = [output_data_cam_13_moving_undistorted; ...
        imagePoints_bv_undistorted{i}(acceptable_detect_id{i}, :)];
end


%% Scatter plot of reprojection errors for every ball size
if make_scatter_plot
    for i = 1:size(imagePoints_a, 2)
        scatter(zeros(size(acceptable_reprojection_errors{i}, 1)) + (ball_size_list(i) + 2*(-0.5 + rand(size(acceptable_reprojection_errors{i}, 1), 1))), ...
            acceptable_reprojection_errors{i}, 10, 'Marker', '.', 'MarkerEdgeColor', 'g')
        hold on
    end
xlabel('Ball diameter (mm)')
ylabel('Reprojection errors (pixels)')
set(gca, 'FontSize', 32)
saveas(gcf, [results_dir, '/figures/reprojection_errors.png'])
end

%% Histogram of reprojection errors
if make_scatter_plot
    figure,
    histogram(acceptable_reprojection_errors_all, 'Normalization', 'probability')
    xlabel('Reprojection errors (pixels)')
    ylabel('Probability')
    set(gca, 'FontSize', 32)
end

%% Export 3-D points with sub-pixel reprojection errors
subpixel_accuracy_id = find(acceptable_reprojection_errors_all < 0.2);
input_data = input_data_moving(subpixel_accuracy_id, :);
output_data_cam_0 = output_data_cam_0_moving(subpixel_accuracy_id, : );
output_data_cam_0_undistorted = output_data_cam_0_moving_undistorted(subpixel_accuracy_id, : );

output_data_cam_1 = output_data_cam_1_moving(subpixel_accuracy_id, :);
output_data_cam_1_undistorted = output_data_cam_1_moving_undistorted(subpixel_accuracy_id, :);
output_data_cam_02 = output_data_cam_02_moving(subpixel_accuracy_id, : );
output_data_cam_02_undistorted = output_data_cam_02_moving_undistorted(subpixel_accuracy_id, : );

output_data_cam_13 = output_data_cam_13_moving(subpixel_accuracy_id, :);
output_data_cam_13_undistorted = output_data_cam_13_moving_undistorted(subpixel_accuracy_id, :);

%% Create structure field for exporting camera parameters
stereoParams_export.TranslationOfCamera2 = stereoParams.TranslationOfCamera2;
stereoParams_export.RotationOfCamera2 = stereoParams.RotationOfCamera2;
stereoParams_export.CameraParameters1K = stereoParams.CameraParameters1.K;
stereoParams_export.CameraParameters2K = stereoParams.CameraParameters2.K;

%% 
crop_x = [270, 975];
crop_x = [90, 1200];
crop_x = [1,1];
lowest_pixel = 195;
stray_pixel_id = find(output_data_cam_0_moving(:,1) > crop_x(1) & output_data_cam_02_moving(:,2) > lowest_pixel...
    & output_data_cam_13_moving(:,2) > lowest_pixel);
input_data = input_data_moving(stray_pixel_id, :);
output_data_cam_0 = output_data_cam_0_moving(stray_pixel_id, : );
output_data_cam_0_undistorted = output_data_cam_0_moving_undistorted(stray_pixel_id, : );

output_data_cam_1 = output_data_cam_1_moving(stray_pixel_id, :);
output_data_cam_1_undistorted = output_data_cam_1_moving_undistorted(stray_pixel_id, :);
output_data_cam_02 = output_data_cam_02_moving(stray_pixel_id, : );
output_data_cam_02_undistorted = output_data_cam_02_moving_undistorted(stray_pixel_id, : );

output_data_cam_13 = output_data_cam_13_moving(stray_pixel_id, :);
output_data_cam_13_undistorted = output_data_cam_13_moving_undistorted(stray_pixel_id, :);

save([results_dir, calibration_target_type, '_data.mat'], 'input_data', 'output_data_cam_0', 'output_data_cam_1', ...
    'output_data_cam_02', 'output_data_cam_13', 'output_data_cam_0_undistorted', 'output_data_cam_02_undistorted', ...
    'output_data_cam_1_undistorted', 'output_data_cam_13_undistorted', 'stereoParams_export');


%% Gather pairwise distances (only applicable for dotted grid)
num_samples = 25000;
repr_err_thresh = 0.25;
if strcmp(calibration_target_type, 'dotted_grid')
    num_images_used = min(500, length(worldPoints));
    [pairwise_distances, output_data_cam_0_pairwise, output_data_cam_1_pairwise, ...
    output_data_cam_02_pairwise, output_data_cam_13_pairwise, ...
    output_data_cam_0_undistorted_pairwise, output_data_cam_1_undistorted_pairwise,...
    output_data_cam_02_undistorted_pairwise, output_data_cam_13_undistorted_pairwise, ...
    acceptable_reprojection_errors_pairwise, ...
    worldPoints_pairwise] = get_pairwise_distances_between_dots(worldPoints, ...
        imagePoints_a, imagePoints_a_undistorted, ...
        imagePoints_b, imagePoints_b_undistorted, ...
        imagePoints_av, imagePoints_av_undistorted, ...
        imagePoints_bv, imagePoints_bv_undistorted, ...
        acceptable_reprojection_errors, ...
        grid_size, grid_spacing, num_images_used);
end

% Filter bad reprojection errors (likely from bad 2-D detections)
bad_idx = any(acceptable_reprojection_errors_pairwise < repr_err_thresh, 2);
idx = randperm(length(pairwise_distances));
idx = setdiff(idx, bad_idx);
idx = idx(1:num_samples);
pairwise_distances = pairwise_distances(idx);
output_data_cam_0_pairwise = output_data_cam_0_pairwise(idx, :);
output_data_cam_1_pairwise = output_data_cam_1_pairwise(idx, :);
output_data_cam_02_pairwise = output_data_cam_02_pairwise(idx, :);
output_data_cam_13_pairwise = output_data_cam_13_pairwise(idx, :);
output_data_cam_0_undistorted_pairwise = output_data_cam_0_undistorted_pairwise(idx, :);
output_data_cam_1_undistorted_pairwise = output_data_cam_1_undistorted_pairwise(idx, :);
output_data_cam_02_undistorted_pairwise = output_data_cam_02_undistorted_pairwise(idx, :);
output_data_cam_13_undistorted_pairwise = output_data_cam_13_undistorted_pairwise(idx, :);
acceptable_reprojection_errors_pairwise = acceptable_reprojection_errors_pairwise(idx, :);
worldPoints_pairwise = worldPoints_pairwise(idx, :);
display('Saving images')
save([results_dir, calibration_target_type, '_pairwise_data.mat'], 'worldPoints_pairwise', 'output_data_cam_0_pairwise', 'output_data_cam_1_pairwise', ...
    'output_data_cam_02_pairwise', 'output_data_cam_13_pairwise', 'output_data_cam_0_undistorted_pairwise', 'output_data_cam_02_undistorted_pairwise', ...
    'output_data_cam_1_undistorted_pairwise', 'output_data_cam_13_undistorted_pairwise', 'stereoParams_export', 'pairwise_distances', 'worldPoints');
display('Results saved')


%%
num_images = length(worldPoints);
num_grid_points = grid_size(1) * grid_size(2);
imagePoints_a_mat = zeros(num_images, num_grid_points, 2);
imagePoints_b_mat = zeros(num_images, num_grid_points, 2);
imagePoints_av_mat = zeros(num_images, num_grid_points, 2);
imagePoints_bv_mat = zeros(num_images, num_grid_points, 2);
imagePoints_a_undistorted_mat = zeros(num_images, num_grid_points, 2);
imagePoints_b_undistorted_mat = zeros(num_images, num_grid_points, 2);
imagePoints_av_undistorted_mat = zeros(num_images, num_grid_points, 2);
imagePoints_bv_undistorted_mat = zeros(num_images, num_grid_points, 2);
worldPoints_mat = zeros(num_images, num_grid_points, 3);

for im_id = 1:num_images
    imagePoints_a_mat(im_id, :, :) = imagePoints_a{im_id};
    imagePoints_b_mat(im_id, :, :) = imagePoints_b{im_id};
    imagePoints_av_mat(im_id, :, :) = imagePoints_av{im_id};
    imagePoints_bv_mat(im_id, :, :) = imagePoints_bv{im_id};
    imagePoints_a_undistorted_mat(im_id, :, :) = imagePoints_a_undistorted{im_id};
    imagePoints_av_undistorted_mat(im_id, :, :) = imagePoints_av_undistorted{im_id};
    imagePoints_b_undistorted_mat(im_id, :, :) = imagePoints_b_undistorted{im_id};
    imagePoints_bv_undistorted_mat(im_id, :, :) = imagePoints_bv_undistorted{im_id};
    worldPoints_mat(im_id, :, :) = worldPoints{im_id};
end

grid = zeros(grid_size(1) * grid_size(2), 3);
for j = 1:grid_size(2)
    for i = 1:grid_size(1)
        grid((i-1)*grid_size(1)+j,:) = [(i - 1) * grid_spacing, (j - 1) * grid_spacing, 0];
    end
end
save([results_dir, calibration_target_type, '_grid_image_data.mat'], 'imagePoints_a_mat', 'imagePoints_b_mat', 'imagePoints_a_undistorted_mat', ...
    'imagePoints_b_undistorted_mat', 'imagePoints_av_mat', 'imagePoints_av_undistorted_mat', 'imagePoints_bv_mat', ...
    'imagePoints_bv_undistorted_mat', 'stereoParams_export', 'worldPoints_mat', 'grid');
display('Results saved')

%% Export 3-D points with sub-pixel reprojection errors: DO NOT USE THIS!
% reprojection_errors_all = [];
% if export_triangulated_points
%     for ball_id = 1:length(ball_size_list)
%         reprojection_errors_all = [reprojection_errors_all; ...
%             reprojection_errors{ball_id}];
%         subpixel_accuracy_id = find(reprojection_errors{ball_id} < 0.5);
%         input_data = [input_data; worldPoints{ball_id}(subpixel_accuracy_id, :)];
%         output_data_cam_0 = [output_data_cam_0; imagePoints_av{ball_id}(subpixel_accuracy_id, :)];
%         output_data_cam_1 = [output_data_cam_1; imagePoints_bv{ball_id}(subpixel_accuracy_id, :)];
%     end
%     save([results_dir, calibration_target_type, '_data.mat'], 'input_data', 'output_data_cam_0', 'output_data_cam_1');
% end
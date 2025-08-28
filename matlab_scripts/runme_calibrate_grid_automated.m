% dataDir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/';
exp_root_folder = dataDir; 
% exp_id = 60;
calibrate_all_cameras = false;
grid_type = 'dot'; % square or dot
cam_names = {'cam_0', 'cam_1', 'cam_2', 'cam_3'};
cam_ids = [1, 2];
grid_data_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/annotations_automated_exp_', num2str(exp_id), '/'];
results_folder = [exp_root_folder, '/exp_', num2str(exp_id)];

if calibrate_all_cameras
    load([grid_data_folder, '/all_cameras/calibration_grid_points_', cam_names{cam_ids(1)}, '.mat'])
else
    load([grid_data_folder, '/calibration_grid_points_', cam_names{cam_ids(1)}, '.mat'])
end
if strcmp(grid_type, 'dot')
    checkerBoardSize = [14, 14] ; %[16, 17]; %[26, 26]; %[14, 18];
    %     checkerBoardSize = [5, 10];
    checkerBoardSquareSize = 1; % mm
else
    checkerBoardSize = [7, 10];
    checkerBoardSquareSize = 0.2; % mm
end

num_images = size(calibration_grid_points, 1);
% results_folder = ['calibration_results_exp_', num2str(exp_id)];

worldPoints = generateCheckerboardPoints(checkerBoardSize + 1, checkerBoardSquareSize);
filter_images = false;


flip_array = {[],...
    [],...
    [],...
    []};
%% Estimate camera parameters

for cam_id = cam_ids
    %     random_idx = randperm(num_images);
    %     random_idx = [1:2, 4:10];
    if calibrate_all_cameras
        load([grid_data_folder, '/all_cameras/calibration_grid_points_', cam_names{cam_id}, '.mat'])
    else
        load([grid_data_folder, '/calibration_grid_points_', cam_names{cam_id}, '.mat'])
    end
    for im_id = 1:num_images
        if (ismember(im_id, flip_array{cam_id}))
            calibration_grid_points(im_id, :, :) = flip(flip(calibration_grid_points(im_id, :, :), 1), 2);
        end
    end
    discard_id = []; %[2,3,4,5,6,7,8,19,30,26,27,31,43,44];
    
    max_grid_images = min(50, size(calibration_grid_points, 1)); % For efficiency
    if  cam_id == 1
        grid_id = 1:size(calibration_grid_points, 1);
        grid_id = randperm(size(calibration_grid_points, 1), max_grid_images);
    end
    % grid_id(discard_id) = [];
    calibration_points_cam = squeeze(calibration_grid_points(grid_id, :, :));
    corners{cam_id} = permute(calibration_points_cam, [2, 3, 1]);
    corners{cam_id}(:,1,:) = corners{cam_id}(:,1,:);
    corners{cam_id}(:,2,:) = corners{cam_id}(:,2,:);
    [cameraParams{cam_id}, imagesUsed, estimationErrors] = estimateCameraParameters(corners{cam_id}, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'mm', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', []);
end
%% Plot camera calibration metrics
for cam_id = cam_ids
    h1=figure; showReprojectionErrors(cameraParams{cam_id});
    h2=figure; showExtrinsics(cameraParams{cam_id}, 'CameraCentric');
    displayErrors(estimationErrors, cameraParams{cam_id});
end

%%
% figure
% for cam_id = 1:2
%     subplot(2,2,cam_id)
%     for i = 1:size(calibration_grid_points, 2)
%         calibration_points_cam = squeeze(calibration_grid_points(cam_id, :, :, :));
%         scatter(squeeze(calibration_points_cam(i, 1)) , ...
%             squeeze(calibration_points_cam(i, 2)), 0.25, 'marker', '.')
%         hold on
%     end
%     title(num2str(cam_names{cam_id}))
%     axis('equal')
%     box on
% end

%%

% for cam_id = 2
%     reprojected_points = cameraParams{cam_id}.ReprojectedPoints;
%     annotated_points = corners{cam_id};
%     for im_id = 1:num_images
%         scatter(squeeze(reprojected_points(:,1,im_id)), squeeze(reprojected_points(:,2,im_id)), 5, 'Marker', '.', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'None');
%         hold on
%         scatter(squeeze(annotated_points(:,1,im_id)), squeeze(annotated_points(:,2,im_id)), 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'None');
%         pause(0.5)
%         hold off
%     end
% end
% axis('equal')

%% Stereo calibration
if length(cam_ids) > 1
    for cam_id_1 = cam_ids(1)
        for cam_id_2 = cam_ids(2)

            imagePoints_a = corners{cam_id_1};
            imagePoints_b = corners{cam_id_2};
            reject_id = [];
            imagePoints_a(:, :, reject_id) = [];
            imagePoints_b(:, :, reject_id) = [];
            imagePoints = cat(4, imagePoints_a, imagePoints_b);

            % Generate world coordinates of the checkerboard keypoints
            worldPoints = generateCheckerboardPoints(checkerBoardSize + 1, checkerBoardSquareSize);

            % Calibrate the camera
            [stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
                'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
                'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'mm', ...
                'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', []);

            % View reprojection errors
            h1=figure; showReprojectionErrors(stereoParams);
            ax = gca;
            ax.FontSize = 24;

            % Visualize pattern locations
            h2=figure; showExtrinsics(stereoParams, 'CameraCentric');
            ax = gca;
            ax.FontSize = 24;

            % Display parameter estimation errors
            displayErrors(estimationErrors, stereoParams);

            % You can use the calibration data to rectify stereo images.
            % I1 = imread(imageFileNames1{1});
            % I2 = imread(imageFileNames2{1});
            % [J1, J2] = rectifyStereoImages(I1, I2, stereoParams);

            % See additional examples of how to use the calibration data.  At the prompt type:
            % showdemo('StereoCalibrationAndSceneReconstructionExample')
            % showdemo('DepthEstimationFromStereoVideoExample')
        end
    end
else
    imagePoints_a = corners{cam_ids(1)};
    cameraParams = cameraParams{cam_ids(1)};
    imagePoints_b = [];
    stereoParams = [];
end

%% Save results
cam_names_used = '';
for cam_id = cam_ids
    cam_names_used = [cam_names_used, cam_names{cam_id}, '_'];
end
mkdir([results_folder, '/results/']);
save([results_folder, '/results/camera_parameters_', cam_names_used(1:end-1), '.mat'], 'cameraParams', 'stereoParams')
save([results_folder, '/results/grid_parameters_', cam_names_used(1:end-1), '.mat'], 'imagePoints_a', 'imagePoints_b', 'worldPoints')


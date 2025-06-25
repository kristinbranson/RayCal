% Script to manually annotate the calibration grid
% Camera ids are cam_0 = 1, cam_1 = 2, reflection of cam_0 = 3, reflection of cam_1 = 4

%% User-defined parameters
% exp_id = 45;
cam_suffix = '';
analyze_all_cams = false;
tilted_cameras = true;
waitTimeBetweenImages = 0;
grid_disp_thresh = 0.1; % (in pixels) This is generally what calibration accuracy is
% dataDir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/';
exp_root_folder = dataDir;
results_folder = fullfile(exp_root_folder, ['/exp_', num2str(exp_id)], '/results', cam_suffix, '/');
mkdir(results_folder)

config_file_name = ['config_exp_ref_', num2str(exp_id)];
cam_names = {'cam_02', 'cam_13'};
if analyze_all_cams
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images_all/'];
    annotations_folder = [exp_root_folder, '/prism/exp_', num2str(exp_id), '/annotations_automated_exp_',num2str(exp_id), '/all_cameras/'];
else
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images/'];
    % calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/raw_data/'];
    annotations_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/annotations_automated_exp_ref_',num2str(exp_id), '/'];
end

%dividing_col = [dividing_col, dividing_col];
%dividing_col = [1067, 1067]; 
% dividing_col = [1241, 1244];

save_individual_grid_coordinates = false;
cam_ids = [1, 2];
grid_size = [9, 9];
num_grid_pts = grid_size(1) * grid_size(2);
use_subset_of_image = false;
num_points = grid_size(1) * grid_size(2);

mkdir(annotations_folder)
if analyze_all_cams
    cam_ids = [1,2,3,4];
end
start_id = 1; % start reading frames from image id 'start_id'
imagePoints_a = []; imagePoints_av = []; imagePoints_b = []; imagePoints_bv = [];
%% Manual annotations
% Initialize parameters

num_cameras = 2 * length(cam_names); % Reflections double the number of cameras available


% Loop over both cameras
num_images = length(dir([calibration_grid_folder, cam_names{1}])) - 2;

%TODO: This is not yet implemented
detection_success = zeros(length(cam_ids), num_images);
rejected_files = cell(length(cam_ids), 1);
accepted_files = cell(num_cameras, 1);
for i = 1:length(cam_ids)
    rejected_files{i} = cell(0,0);
end

% Make sure all cameras have recorded the same number of images
calibration_grid_image_paths = [];
for cam_id = cam_ids
    calibration_grid_image_paths{cam_id} = dir(fullfile([calibration_grid_folder, cam_names{cam_id}], '*.png'));
    try 
        lengths = [];
        for cam_id = cam_ids
            lengths = [lengths; length(calibration_grid_image_paths{cam_id})];
        end
        if all(lengths == lengths(1))
            disp('Check passed: All cameras have the same number of frames')
        else
            errors('Number of images collected across different cameras do not match')
        end
    catch ME
        disp('Error: ', Me.message)
    end
end

%%
for cam_id = cam_ids
    imagePoints_r_prev = [];
    imagePoints_v_prev = [];
    grid_disp_r = inf; % initializing for the first frame
    grid_disp_v = inf; % initializing for the first frame
    num_images = length(dir([calibration_grid_folder, cam_names{cam_id}])) - 2;
    calibration_grid_points = zeros(num_images, num_points, 2);
    
    % Read image of a grid orientation for the camera
    image_names = {};
    for im_id = start_id:length(calibration_grid_image_paths{cam_id}) - 2
        im = (imread([calibration_grid_folder, cam_names{cam_id}, '/', calibration_grid_image_paths{cam_id}(im_id).name]));
        image_names = [image_names; calibration_grid_image_paths{cam_id}(im_id).name];
        display(['Analyzing ', calibration_grid_image_paths{cam_id}(im_id).name, ' image id: ', num2str(im_id), ' of cam ', cam_names{cam_id}])

        if cam_id == 1
            im_v = im(:, 1:dividing_col(cam_id));
            im_r = im(:, dividing_col(cam_id) + 1: end);
        else
            im_v = im(:, 1:dividing_col(cam_id));
            im_r = im(:, dividing_col(cam_id) + 1: end);
        end

        %         im_r = adapthisteq(im_r);
        %         im_v = adapthisteq(im_v);

        [imagePoints_r, ~] = detectCircleGridPoints(im_r, grid_size, 'PatternType', 'symmetric');
        [imagePoints_v, ~] = detectCircleGridPoints(im_v, grid_size, 'PatternType', 'symmetric');
        if size(imagePoints_r, 1) ~= num_grid_pts || size(imagePoints_v, 1) ~= num_grid_pts
            rejected_files{cam_id} = [rejected_files{cam_id}; calibration_grid_image_paths{cam_id}(im_id).name];
            continue
        end
        %TODO: Implement this with an if else statement
        %detection_success[cam_id, im_id] = True;
        %%
        imagePoints_r = rectify_order(imagePoints_r, grid_size);
        imagePoints_v = rectify_order(imagePoints_v, grid_size);
        %%

        if cam_id == 1 || cam_id == 2
            imagePoints_r = flip_grid_v(imagePoints_r, grid_size);
        else
            imagePoints_v = flip_grid_v(imagePoints_v, grid_size);
        end
        
        smallest_dist_r = diff(imagePoints_r, 1);
        smallest_dist_r = min(sqrt(smallest_dist_r(:,1).^2 + smallest_dist_r(:,2).^2));
        smallest_dist_v = diff(imagePoints_v, 1);
        smallest_dist_v = min(sqrt(smallest_dist_v(:,1).^2 + smallest_dist_v(:,2).^2));
        %%
        if smallest_dist_r < 5 || smallest_dist_v < 5 % TODO: This should be set by the size of a dot in pixels on the calibration grid 
            rejected_files{cam_id} = [rejected_files{cam_id}; calibration_grid_image_paths{cam_id}(im_id).name];
            continue
        end

        if im_id > 1 && ~isempty(imagePoints_r_prev) && ~isempty(imagePoints_v_prev)
            grid_disp_r = mean(vecnorm(imagePoints_r - imagePoints_r_prev, 2, 2));
            grid_disp_v = mean(vecnorm(imagePoints_v - imagePoints_v_prev, 2, 2));
        end

        if grid_disp_r < grid_disp_thresh || grid_disp_v < grid_disp_thresh
            rejected_files{cam_id} = [rejected_files{cam_id}; calibration_grid_image_paths{cam_id}(im_id).name];
            continue
        else
            accepted_files{cam_id} = [accepted_files{cam_id}; im_id];
            imagePoints_r_prev = imagePoints_r;
            imagePoints_v_prev = imagePoints_v;
        end

        %         if tilted_cameras && cam_id == 2 % One of the cameras was rotated wrt the other
        %             imagePoints_r = flip_grid_h(imagePoints_r, grid_size);
        %             imagePoints_v = flip_grid_h(imagePoints_v, grid_size);
        %         end

        if cam_id == 1
            imagePoints_r(:,1) = imagePoints_r(:,1) + dividing_col(cam_id);
        else
            imagePoints_r(:,1) = imagePoints_r(:,1) + dividing_col(cam_id);
        end

        if cam_id == 1
            imagePoints_a{im_id} = imagePoints_r;
            imagePoints_av{im_id} = imagePoints_v;
        elseif cam_id == 2
            imagePoints_b{im_id} = imagePoints_r;
            imagePoints_bv{im_id} = imagePoints_v;
        end

        % Save all {num_point} coordinates for the grid orientation and
        % camera as a mat file
        im_calibration_grid = calibration_grid_points(im_id, :, :);
        %         save([annotations_folder, 'im_', num2str(im_id), '_cam_', num2str(cam_id), '.mat'], 'im_calibration_grid')

        J = insertText(im,imagePoints_r,1:size(imagePoints_r,1), 'FontSize', 12, 'BoxOpacity', 0);
        J = insertMarker(J,imagePoints_r, 'x', 'color', 'green', Size=6);
        J = insertText(J,imagePoints_v,1:size(imagePoints_v,1), 'FontSize', 12, 'BoxOpacity', 0);
        J = insertMarker(J,imagePoints_v, 'x', 'color', 'green', Size=6);
        imshow(J)
        title("Detected a Circle Grid of Dimensions " + mat2str(num_points))
        drawnow
        pause(waitTimeBetweenImages)
    end

    if ~use_subset_of_image
        crop_coor = [];
    end

    save_config_file(config_file_name, calibration_grid_folder, annotations_folder, cam_names, cam_id, ...
        crop_coor, image_names)
end
%%
filtered_points = intersect(accepted_files{1}, accepted_files{2});
for i = 1:length(filtered_points)
    imagePoints_a_filtered{i} = imagePoints_a{filtered_points(i)};
    imagePoints_av_filtered{i} = imagePoints_av{filtered_points(i)};
    imagePoints_b_filtered{i} = imagePoints_b{filtered_points(i)};
    imagePoints_bv_filtered{i} = imagePoints_bv{filtered_points(i)};
end

imagePoints_a = imagePoints_a_filtered;
imagePoints_av = imagePoints_av_filtered;
imagePoints_b = imagePoints_b_filtered;
imagePoints_bv = imagePoints_bv_filtered;
% Remove  all missed detections from both cameras
save([results_folder, 'grid_points_ref.mat'], 'imagePoints_a', 'imagePoints_av', ...
    'imagePoints_b', 'imagePoints_bv')

%%
% imagePoints_v = transpose_grid(imagePoints_v, grid_size);
%%


%% Save all annotations
% save([annotations_folder, 'calibration_grid_points.mat'], 'calibration_grid_points')

function imagePoints = flip_grid_h(imagePoints, grid_size)
id = 1:size(imagePoints, 1);
id = reshape(id, grid_size);
id = flipud(id);
id = id(:);
imagePoints = imagePoints(id, :);
end

function imagePoints = flip_grid_v(imagePoints, grid_size)
id = 1:size(imagePoints, 1);
id = reshape(id, grid_size);
id = fliplr(id);
id = id(:);
imagePoints = imagePoints(id, :);
end

function imagePoints = transpose_grid(imagePoints, grid_size)
% NOTE: The tranpose is not about the diagonal o the matrix
id = 1:size(imagePoints, 1);
id = reshape(id, grid_size);
id = id';
id = id(:);
imagePoints = imagePoints(id, :);
end

function imagePoints = rectify_order(imagePoints, grid_size)
if imagePoints(1,1) < imagePoints(end,1) && imagePoints(1,2) < imagePoints(end,2)
    if imagePoints(grid_size(1), 1) > imagePoints(grid_size(1) * (grid_size(2)-1) + 1, 1)
        imagePoints = transpose_grid(imagePoints, grid_size);
        imagePoints = flip_grid_v(imagePoints, grid_size);
        imagePoints = flip_grid_h(imagePoints, grid_size);
    end
end

if imagePoints(1,1) > imagePoints(end,1) && imagePoints(1,2) > imagePoints(end,2)
    if imagePoints(grid_size(1), 1) < imagePoints(grid_size(1) * (grid_size(2)-1) + 1, 1)
        imagePoints = transpose_grid(imagePoints, grid_size);
    else
        imagePoints = flip_grid_v(imagePoints, grid_size);
        imagePoints = flip_grid_h(imagePoints, grid_size);
    end
end

if imagePoints(1,1) > imagePoints(end,1) && imagePoints(1,2) < imagePoints(end,2)
    if imagePoints(grid_size(1), 1) < imagePoints(grid_size(1) * (grid_size(2)-1) + 1, 1)
        imagePoints = transpose_grid(imagePoints, grid_size);
    end
    imagePoints = flip_grid_v(imagePoints, grid_size);
end

if imagePoints(1,1) < imagePoints(end,1) && imagePoints(1,2) > imagePoints(end,2)
    if imagePoints(grid_size(1), 1) > imagePoints(grid_size(1) * (grid_size(2)-1) + 1, 1)
        imagePoints = transpose_grid(imagePoints, grid_size);
    end
    imagePoints = flip_grid_h(imagePoints, grid_size);
end

end

% Script to manually annotate the calibration grid
% Camera ids are cam_0 = 1, cam_1 = 2, reflection of cam_0 = 3, reflection of cam_1 = 4

%% User-defined parameters
% exp_id = 23;
makeVideo = false;
analyze_all_cams = false;
tilted_cameras = true;
waitTimeBetweenImages = 0;
grid_disp_thresh = 0.1; % (in pixels) This is generally what calibration accuracy is 
exp_root_folder = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/';

config_file_name = ['config_exp_', num2str(exp_id)];
cam_names = {'cam_0', 'cam_1', 'cam_2', 'cam_3'};
if analyze_all_cams
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images_all/'];
    annotations_folder = [exp_root_folder, '/prism/exp_', num2str(exp_id), '/annotations_automated_exp_',num2str(exp_id), '/all_cameras/'];
else
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images/'];
    % calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/raw_data/'];
    annotations_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/annotations_automated_exp_',num2str(exp_id), '/'];
end

save_individual_grid_coordinates = false;
cam_ids = [1, 2];
dividing_row = 295;
grid_size = [16, 17]; %[26, 26]; % [14, 18]; 
num_grid_pts = grid_size(1) * grid_size(2);
use_subset_of_image = false;
num_points = grid_size(1) * grid_size(2);

mkdir(annotations_folder)
if analyze_all_cams
    cam_ids = [1,2];
end
rejected_files = cell(length(cam_ids), 1);
accepted_files = cell(length(cam_ids), 1);
for i = 1:length(cam_ids)
    rejected_files{i} = cell(0,0);
end
start_id = 1; % start reading frames from image id 'start_id'
%% Manual annotations
% Initialize parameters

num_cameras = 2 * length(cam_names); % Reflections double the number of cameras available
if makeVideo
    v = VideoWriter('calibration_movie.avi');
    open(v);
end
% Loop over both cameras
for cam_id = cam_ids
    imagePoints_prev = [];
    grid_disp = inf;
    calibration_grid_image_paths = [];
    num_images = length(dir([calibration_grid_folder, cam_names{cam_id}])) - 2;
    calibration_grid_points_{cam_id} = zeros(num_images, num_points, 2);
    calibration_grid_image_paths{cam_id} = dir(fullfile([calibration_grid_folder, cam_names{cam_id}], '*.png'));

    % Read image of a grid orientation for the camera
    image_names = {};
    for im_id = start_id:length(calibration_grid_image_paths{cam_id}) - 2
        im = (imread([calibration_grid_folder, cam_names{cam_id}, '/', calibration_grid_image_paths{cam_id}(im_id).name]));

        image_names = [image_names; calibration_grid_image_paths{cam_id}(im_id).name];
        display(['Analyzing ', calibration_grid_image_paths{cam_id}(im_id).name, ' image id: ', num2str(im_id), ' of cam ', cam_names{cam_id}])
%         close
        if (use_subset_of_image)
            if cam_id <= 2
                crop_coor = [1,dividing_row,1,size(im,2)];
            else
                crop_coor = [dividing_row,size(im,1),1,size(im,2)];
            end
        else
            crop_coor = [1,size(im,1),1,size(im,2)];
        end
        im = im(crop_coor(1):crop_coor(2), crop_coor(3):crop_coor(4));
        [imagePoints, ~] = detectCircleGridPoints(im, grid_size, 'PatternType', 'symmetric');
        
        

        if size(imagePoints, 1) ~= num_grid_pts
            rejected_files{cam_id} = [rejected_files{cam_id}; calibration_grid_image_paths{cam_id}(im_id).name];
            continue
        end

        if im_id > 1 && ~isempty(imagePoints_prev)
            grid_disp = mean(vecnorm(imagePoints - imagePoints_prev, 2, 2));
        end

        if grid_disp < grid_disp_thresh
            rejected_files{cam_id} = [rejected_files{cam_id}; calibration_grid_image_paths{cam_id}(im_id).name];
            continue
        else
            accepted_files{cam_id} = [accepted_files{cam_id}; im_id];
            imagePoints_prev = imagePoints;
        end

%         if (imagePoints(1,2) > imagePoints(2,2))
%             imagePoints = flip(imagePoints, 1);
%         end
        imagePoints = rectify_order(imagePoints, grid_size);
        
        if tilted_cameras && cam_id == 2 % One of the cameras was rotated wrt the other 
            imagePoints = flipud((imagePoints));
        end
        if cam_id == 2
            imagePoints = flip(imagePoints);
        end
        
        calibration_grid_points_{cam_id}(im_id, :, :) = imagePoints;

        % Save all {num_point} coordinates for the grid orientation and
        % camera as a mat file
        im_calibration_grid = calibration_grid_points_{cam_id}(im_id, :, :);
        save([annotations_folder, 'im_', num2str(im_id), '_cam_', num2str(cam_id), '.mat'], 'im_calibration_grid')
        
        J = insertText(im,imagePoints,1:size(imagePoints,1), 'FontSize', 12, 'BoxOpacity', 0);
        J = insertMarker(J,imagePoints, 'x', 'color', 'green', Size=6);
        imshow(J)
        title("Detected a Circle Grid of Dimensions " + mat2str(num_points))
        drawnow
        if makeVideo
            f = getframe;
            writeVideo(v, f)
        end
        pause(waitTimeBetweenImages)
    end
end

for cam_id = cam_ids
    filtered_points = intersect(accepted_files{1}, accepted_files{2});
    calibration_grid_points_filtered = zeros(length(filtered_points), ...
    num_grid_pts, ...
    2);
    for i = 1:length(filtered_points)
        calibration_grid_points_filtered(i, :, :) = calibration_grid_points_{cam_id}(filtered_points(i), :, :);
    end
    calibration_grid_points_{cam_id} = calibration_grid_points_filtered;
    calibration_grid_points = calibration_grid_points_{cam_id};
    save([annotations_folder, 'calibration_grid_points_', cam_names{cam_id}, '.mat'], 'calibration_grid_points')   
    if ~use_subset_of_image
        crop_coor = [];
    end
    save_config_file(config_file_name, calibration_grid_folder, annotations_folder, cam_names, cam_id, ...
        crop_coor, image_names)
end

if makeVideo
    close(v)
end
%% Save all annotations
% save([annotations_folder, 'calibration_grid_points.mat'], 'calibration_grid_points')
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

function imagePoints = transpose_grid(imagePoints, grid_size)
% NOTE: The tranpose is not about the diagonal o the matrix
    id = 1:size(imagePoints, 1);
    id = reshape(id, grid_size);
    id = id';
    id = id(:);
    imagePoints = imagePoints(id, :);
end

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
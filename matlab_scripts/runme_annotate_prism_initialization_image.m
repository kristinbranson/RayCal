%% User-defined parameters
% exp_id = 25;
prism_size = 20; % mm
cam_suffix = '';
analyze_all_cams = false;
tilted_cameras = true;
waitTimeBetweenImages = 0;
exp_root_folder = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/';
results_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/results', cam_suffix, '/'];
mkdir(results_folder)

config_file_name = ['config_exp_ref_', num2str(exp_id)];
cam_names = {'cam_02'};
if analyze_all_cams
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images_all/'];
    annotations_folder = [exp_root_folder, '/prism/exp_', num2str(exp_id), '/annotations_automated_exp_',num2str(exp_id), '/all_cameras/'];
else
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images/'];
    % calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/raw_data/'];
    annotations_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/annotations_automated_exp_ref_',num2str(exp_id), '/'];
end

dividing_col = [1155, 1155]; 

save_individual_grid_coordinates = false;
cam_ids = [1];
grid_size = [9, 9];
use_subset_of_image = false;
num_points = grid_size(1) * grid_size(2);

mkdir(annotations_folder)
if analyze_all_cams
    cam_ids = [1,2,3,4];
end
start_id = 1; % start reading frames from image id 'start_id'
imagePoints_a = []; imagePoints_av = []; imagePoints_b = []; imagePoints_bv = [];
im_id = 1;
cams_used_for_calibration = 'cam_0_cam_1';
load([results_folder, '/camera_parameters_', cams_used_for_calibration, '.mat'])

if length(cameraParams) == 2
    cameraParams = cameraParams{1};
end

checkerBoardSize = [9, 9];
checkerBoardSquareSize = 1;
worldPoints = generateCheckerboardPoints(checkerBoardSize + 1, checkerBoardSquareSize);
worldPoints = [worldPoints, zeros(size(worldPoints,1), 1)];

%% Manual annotations
% Initialize parameters
num_cameras = 2 * length(cam_names); % Reflections double the number of cameras available
if exist([calibration_grid_folder, 'initialization.png'])
    im = imread([calibration_grid_folder, 'initialization.png']);
elseif exist([calibration_grid_folder, 'initialization.bmp'])
    im = imread([calibration_grid_folder, 'initialization.bmp']);
end

for cam_id = cam_ids
    if cam_id == 1
        im_r = im(:, dividing_col(cam_id) + 1: end);
    else
        im_r = im(:, 1:dividing_col(cam_id));
    end
    
    [imagePoints_r, ~] = detectCircleGridPoints(im_r, grid_size, 'PatternType', 'symmetric');

    imagePoints_r = rectify_order(imagePoints_r, grid_size);
%%      

    if cam_id == 1
        imagePoints_r = flip_grid_v(imagePoints_r, grid_size);
    end


    if tilted_cameras && cam_id == 2 % One of the cameras was rotated wrt the other
        imagePoints_r = flip_grid_h(imagePoints_r, grid_size);
    end

    if cam_id == 1
        imagePoints_r(:,1) = imagePoints_r(:,1) + dividing_col(cam_id);
    end
    
    if cam_id == 1
        imagePoints_a{im_id} = imagePoints_r;
    elseif cam_id == 2
        imagePoints_b{im_id} = imagePoints_r;
    end

    % Save all {num_point} coordinates for the grid orientation and
    % camera as a mat file
%         save([annotations_folder, 'im_', num2str(im_id), '_cam_', num2str(cam_id), '.mat'], 'im_calibration_grid')

    J = insertText(im,imagePoints_r,1:size(imagePoints_r,1), 'FontSize', 12, 'BoxOpacity', 0);
    J = insertMarker(J,imagePoints_r, 'x', 'color', 'green', Size=6);
    imshow(J)
    title("Detected a Circle Grid of Dimensions " + mat2str(num_points))
    drawnow
end

%% Get camera pose wrt the grid
imagePoints_r_undistorted = undistortPoints(imagePoints_r, cameraParams);
[worldOrientation, worldLocation] = estimateWorldCameraPose(imagePoints_r_undistorted, worldPoints, cameraParams);
figure,
scatter3(worldPoints(:,1), worldPoints(:,2), worldPoints(:,3))
hold on
plotCamera('Size', 2, 'Orientation', worldOrientation, 'Location',...
worldLocation);
worldOrientation = worldOrientation';
axis equal
xlabel('X (mm)', 'FontSize', 32)
ylabel('Y (mm)', 'FontSize', 32)
zlabel('Z (mm)', 'FontSize', 32)
set(gca,'FontSize', 26)
box on 
vec = [0,0,1] * worldOrientation';
vec = norm(worldLocation) * [0, 0, 0; vec];
vec = vec + worldLocation;
plot3(vec(:,1), vec(:,2), vec(:,3))
hold off

figure,
axes_grid = eye(3);
worldPoints_rot = (worldPoints - worldLocation) * worldOrientation;
origin = [0,0,0];
origin = (origin - worldLocation) * worldOrientation;
worldOrientation_cam_frame = eye(3);
worldLocation_cam_frame = [0, 0, 0];
scatter3(worldPoints_rot(:,1), worldPoints_rot(:,2), worldPoints_rot(:,3), 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k')
hold on
plotCamera('Size', 2, 'Orientation', worldOrientation_cam_frame, 'Location',...
worldLocation_cam_frame);
axis equal
xlabel('X (mm)', 'FontSize', 32)
ylabel('Y (mm)', 'FontSize', 32)
zlabel('Z (mm)', 'FontSize', 32)
set(gca,'FontSize', 26)
box on 
hold off
view([0, 70])
camroll(90)

%% Compute grid axis

horizontal_id = 2;
vertical_id = grid_size(1) + 1;
axes_grid(:,2) = -(worldPoints_rot(1,:) - worldPoints_rot(horizontal_id, :))';
axes_grid(:,3) = (worldPoints_rot(1,:) - worldPoints_rot(vertical_id, :))';
axes_grid(:,1) = cross(axes_grid(:,2), axes_grid(:,3));
axes_grid = axes_grid ./ repmat(vecnorm(axes_grid), [3,1]);
hold on 
for i = 1:3
    plot3([worldPoints_rot(1,1), worldPoints_rot(1,1) + axes_grid(1,i)], ...
        [worldPoints_rot(1,2), worldPoints_rot(1,2) + axes_grid(2,i)], ...
        [worldPoints_rot(1,3), worldPoints_rot(1,3) + axes_grid(3,i)])
end
grid_location = mean(worldPoints_rot, 1);

%% Compute prism axis
% grid_offset = (grid_size(1) + 2) / sqrt(2) / 2;
grid_offset = (grid_size(1) + 0.5) / sqrt(2) / 2;
% grid_mount_height = 3;
grid_mount_height = 3.5;
% axes_prism = roty(-45) * axes_grid;
axes_prism = rotate_vector(axes_grid, axes_grid(:,2), -45); % Rotate about the horizontal axis
location_prism = grid_location + (prism_size - grid_offset) * axes_prism(:,1)' - (prism_size / 2 + grid_offset + grid_mount_height) * axes_prism(:,3)';
save([results_folder, '/prism_initialization.mat'], 'location_prism', 'axes_prism')

%% Show all axes
plot_axes(axes_grid, grid_location, 'k')
axes_prism1 = roty(-45) * axes_grid;
location_prism = grid_location + (prism_size - grid_offset) * axes_prism1(:,1)' - (prism_size / 2 + grid_offset + grid_mount_height) * axes_prism1(:,3)';
plot_axes(axes_prism1, location_prism, 'r')
axes_prism2 = rotate_vector(axes_grid, axes_grid(:,2), -45);
location_prism = grid_location + (prism_size - grid_offset) * axes_prism2(:,1)' - (prism_size / 2 + grid_offset + grid_mount_height) * axes_prism2(:,3)';
plot_axes(axes_prism2, location_prism, 'g')

%%
temp = rotate_vector(axes_grid, [0;1;0], -45);
%%
function [] = plot_axes(axes_, origin, color)
    plot3(origin(1) + [0, axes_(1,1)], origin(2) + [0, axes_(2,1)], origin(3) + [0, axes_(3,1)],...
        'Linewidth', 3, 'color', color)
    hold on
    plot3(origin(1) + [0, axes_(1,2)], origin(2) + [0, axes_(2,2)], origin(3) + [0, axes_(3,2)],...
        'color', color)
    plot3(origin(1) + [0, axes_(1,3)], origin(2) + [0, axes_(2,3)], origin(3) + [0, axes_(3,3)],...
        'color', color)
    axis('equal')
    grid on
    box on
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
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

%% Rodrigues rotation copied from https://www.mathworks.com/matlabcentral/fileexchange/73828-rodrigues-axis-angle-rotation/
function v_rot = rotate_vector(v_in,k,theta)
    v_rot = zeros(3,3);
    k = k';
    for i = 1:3
        v = v_in(:,i)';
        v_rot(:,i) = v*cosd(theta)+cross(k,v)*sind(theta)+k*(dot(k,v))*(1-cosd(theta));
    end
end

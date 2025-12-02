exp_root_folder = dataDir;
opengl('save', 'software');

analyze_all_cams = false;
if analyze_all_cams
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images_all/'];
    annotations_folder = [exp_root_folder, '/prism/exp_', num2str(exp_id), '/annotations_automated_exp_',num2str(exp_id), '/all_cameras/'];
else
    calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/calibration_grid_images/'];
    % calibration_grid_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/raw_data/'];
    annotations_folder = [exp_root_folder, '/exp_', num2str(exp_id), '/annotations_automated_exp_ref_',num2str(exp_id), '/'];
end

% Initialize parameters
if exist([calibration_grid_folder, 'initialization_cam_0.png'])
    im = imread([calibration_grid_folder, 'initialization_cam_0.png']);
elseif exist([calibration_grid_folder, 'initialization_cam_0.bmp'])
    im = imread([calibration_grid_folder, 'initialization_cam_0.bmp']);
end

figure,
imshow(im)
display('Click on any point on the line separating real view and virtual view')
[x,y] = ginput(1);
display(['Dividing column set to ', num2str(round(x))])
close all
dividing_col(:,1) = round(x);
image_width(:,1) = size(im, 2);

if exist([calibration_grid_folder, 'initialization_cam2.png'])
    im = imread([calibration_grid_folder, 'initialization_cam2.png']);
elseif exist([calibration_grid_folder, 'initialization_cam2.bmp'])
    im = imread([calibration_grid_folder, 'initialization_cam2.bmp']);
end
figure,
imshow(im)
display('Click on any point on the line separating real view and virtual view')
[x,y] = ginput(1);
display(['Dividing column set to ', num2str(round(x))])
close all
dividing_col(:,2) = round(x);
image_width(:,2) = size(im, 2);

%dividing_col = [1229, 1229];
fileID = fopen([calibration_grid_folder, '/logfile.txt'], 'w');
fprintf(fileID, '%d\n', round(dividing_col(:,1)));
fprintf(fileID, '%d\n', round(dividing_col(:,2)));
fprintf(fileID, '%d\n', round(image_width(:,1)));
fprintf(fileID, '%d\n', round(image_width(:,2)));
fclose(fileID);


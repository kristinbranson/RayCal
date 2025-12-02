%% Save config file
function save_config_file(config_file_name, calibration_grid_folder, annotations_folder,...
    cam_names, cam_id, crop_coor, image_names)
config.calibration_grid_folder = calibration_grid_folder;
config.annotations_folder = annotations_folder;
config.cam_id = cam_id;
config.crop_coor = crop_coor;
config.image_names = image_names;
save([annotations_folder, '/', config_file_name, '_', cam_names{cam_id}, '.mat'], 'config')
end
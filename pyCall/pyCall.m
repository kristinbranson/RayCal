model_path = 'model_single_camera_prism.pth';
pyenv('Version','/groups/branson/bransonlab/aniket/pytorch_remote/bin/python')
epipolar_line = pyrunfile("return_projected_ray_single_camera_prism.py", ...
    "epipolar_line", user_annotation=[400,500], PATH=model_path);
epipolar_line = double(epipolar_line);

%%
model_path = 'model_two_cameras_prism.pth';
pyenv('Version','/groups/branson/bransonlab/aniket/pytorch_local/bin/python')
cam_label = "primary_virtual";
[epipolar_line_unlabelled, epipolar_line_unlabelled] = pyrunfile("return_projected_ray_two_cameras_prism.py",...
    "epipolar_line_unlabelled", "epipolar_line_unlabelled", ...
    user_annotation=[400,500], PATH=model_path, cam_label=cam_label);
epipolar_line_unlabelled = double(epipolar_line_1);
epipolar_line_labelled = double(epipolar_line_2);
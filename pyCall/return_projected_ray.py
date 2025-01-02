import matplotlib.pyplot as plt
import numpy as np
import torch
from arenasEfficient import Arena_single_camera_prism_grid_distance
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import openpyxl
import sys

#%%
# Load  data
#model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2024_12_26_12_53_16'
#PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
PATH = 'model.pth'
print(PATH)
checkpoint = torch.load(PATH, weights_only=True)
arena = Arena_single_camera_prism_grid_distance(
            principal_point_pixel_cam_0=torch.tensor([0.,0.]).to(torch.float64),
            principal_point_pixel_cam_1=torch.tensor([0.,0.]).to(torch.float64), 
            focal_length_cam_0=torch.tensor(0.).to(torch.float64), 
            focal_length_cam_1=torch.tensor(0.).to(torch.float64),
            R_stereo_cam=None,
            T_stereo_cam=None, 
            prism_angles=torch.tensor([0.,0.,0.]).to(torch.float64),
            prism_center=torch.tensor([0.,0.,0.]).to(torch.float64).unsqueeze(-1),
            prism_size=20.,
) 
arena.load_state_dict(checkpoint)
experiment_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_17/fly_images/cam_0/'
images_dir = [os.path.join(experiment_dir, filename) for filename in os.listdir(experiment_dir) if filename.startswith("image")][0]

# %%
def get_annotations_curve(arena, user_annotation):
    with torch.no_grad():
        undistorted_annotations = arena.camera1.undistort_pixels_classical(user_annotation[:2],
        arena.radial_dist_coeffs)
        cam_1_ray = arena.camera1(undistorted_annotations)
        _, _, emergent_ray, _ = arena.prism(cam_1_ray)
        origin = emergent_ray.origin[:,0][:,None]
        direction = emergent_ray.direction[:,0][:,None]
        s = torch.linspace(0, 3, 100).to(torch.float64)
        r = origin + s[None, :] * direction

        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        annotations_curve = arena.camera1.reproject(torch.tensor(r).to(torch.float64), R1, T1)
        annotations_curve = arena.camera1.distort_pixels_classical(annotations_curve,
        arena.radial_dist_coeffs)
        return annotations_curve

# user_annotation = sys.argv[0]
user_annotation = torch.tensor(user_annotation)
epipolar_line = get_annotations_curve(arena, user_annotation).numpy()
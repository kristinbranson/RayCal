import matplotlib.pyplot as plt
import numpy as np
import torch
from arenasEfficient import Arena_reprojection_loss_two_cameras_prism_grid_distances
import scipy.io as sio
import os
from matplotlib.widgets import Cursor
import openpyxl
import sys
from ray_tracing_simulator_nnModules_grad import get_rot_mat

#%%
# Load  data
#model_checkpoint_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/calprism/outputs/model_checkpoints/2024_12_26_12_53_16'
#PATH = f'{model_checkpoint_dir}/best_checkpoint.pth'
# PATH = 'model.pth'
print(f'Loading model from: {PATH}')
checkpoint = torch.load(PATH, weights_only=True)
arena = Arena_reprojection_loss_two_cameras_prism_grid_distances(
            principal_point_pixel_cam_0=torch.tensor([0.,0.]).to(torch.float64),
            principal_point_pixel_cam_1=torch.tensor([0.,0.]).to(torch.float64), 
            focal_length_cam_0=torch.tensor(0.).to(torch.float64), 
            focal_length_cam_1=torch.tensor(0.).to(torch.float64),
            R_stereo_cam=torch.ones(3,3).to(torch.float64),
            T_stereo_cam=torch.ones(3,1).to(torch.float64), 
            prism_angles=torch.tensor([0.,0.,0.]).to(torch.float64),
            prism_center=torch.tensor([0.,0.,0.]).to(torch.float64).unsqueeze(-1),
            ) 
arena.load_state_dict(checkpoint)
experiment_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism_new_led/exp_17/fly_images/cam_0/'
images_dir = [os.path.join(experiment_dir, filename) for filename in os.listdir(experiment_dir) if filename.startswith("image")][0]

# %%
#NOTE: This is wrong. Needs to be fixed. Start with providing the correct R and T 
def get_annotations_curve(arena, user_annotation, 
                          camera_real, cam_real_dist_coeff,
                          camera_virtual, cam_virtual_dist_coeff):
    with torch.no_grad():
        undistorted_annotations = camera_virtual.undistort_pixels_classical(user_annotation[:2],
        cam_real_dist_coeff)
        cam_1_ray = camera_virtual(undistorted_annotations)
        _, _, emergent_ray, _ = arena.prism(cam_1_ray)
        origin = emergent_ray.origin[:,0][:,None]
        direction = emergent_ray.direction[:,0][:,None]
        s = torch.linspace(0, 3, 100).to(torch.float64)
        r = origin + s[None, :] * direction

        R1 = torch.eye(3, 3).to(torch.float64)
        T1 = torch.zeros(3, 1).to(torch.float64)
        R_stereo_cam = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            )
        
        annotations_curve_real = camera_real.reproject(torch.tensor(r).to(torch.float64), R1, T1)
        annotations_curve_virtual = camera_virtual.reproject(torch.tensor(r).to(torch.float64), R_stereo_cam, arena.T_stereo_cam)
        annotations_curve_real = camera_real.distort_pixels_classical(
                                                            annotations_curve_real,
                                                            cam_real_dist_coeff)
        annotations_curve_virtual = camera_virtual.distort_pixels_classical(
                                                            annotations_curve_virtual,
                                                            cam_virtual_dist_coeff)
        return annotations_curve_real, annotations_curve_virtual

def get_epipolar_line(arena, user_annotation, 
                          cam_label):
    """
    Input: user_annotation: pixels coordinates of the user annotation (tensor)
    cam_label: label of the camera whose annotation is provided by the user ["primary_virtual", "primary_real", "secondary_virtual", "secondary_real"]
    Output:
    If cam_label is "primary_virtual" or "secondary_virtual", returns the annotations of the curve in the unlabelled camera and the labelled camera
    If cam_label is "primary_real" or "secondary_real", returns the annotations of the curve in the unlabelled camera
    """
    if "primary" in cam_label:
        labelled_camera = arena.camera1
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_0
        unlabelled_camera = get_secondary_camera(arena)
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        R_labelled = torch.eye(3, 3).to(torch.float64)
        T_labelled = torch.zeros(3, 1).to(torch.float64)
        R_unlabelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_unlabelled = arena.T_stereo_cam

    elif "secondary" in cam_label:
        labelled_camera = get_secondary_camera(arena)
        cam_labelled_dist_coeff = arena.radial_dist_coeffs_cam_1
        unlabelled_camera = arena.camera1
        cam_unlabelled_dist_coeff = arena.radial_dist_coeffs_cam_0    
        R_unlabelled = torch.eye(3, 3).to(torch.float64)
        T_unlabelled = torch.zeros(3, 1).to(torch.float64)
        R_labelled = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            ) 
        T_labelled = arena.T_stereo_cam

    else:
        print("Please provide appropriate camera labels")


    if cam_label in ["primary_virtual", "secondary_virtual"]:
        with torch.no_grad():
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            cam_1_ray = labelled_camera(undistorted_annotations)
            _, _, emergent_ray, _ = arena.prism(cam_1_ray)
            origin = emergent_ray.origin[:,0][:,None]
            direction = emergent_ray.direction[:,0][:,None]
            s = torch.linspace(0, 8, 50).to(torch.float64)
            r = origin + s[None, :] * direction        
            
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(torch.tensor(r).to(torch.float64), R_unlabelled, T_unlabelled)
            annotations_curve_labelled_camera = labelled_camera.reproject(torch.tensor(r).to(torch.float64), R_labelled, T_labelled)
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            annotations_curve_labelled_camera = labelled_camera.distort_pixels_classical(
                                                                annotations_curve_labelled_camera,
                                                                cam_labelled_dist_coeff)
            return annotations_curve_unlabelled_camera, annotations_curve_labelled_camera
    

    elif cam_label in ["primary_real", "secondary_real"]:
        with torch.no_grad():
            prism_distance = torch.norm(arena.prism_center)
            undistorted_annotations = labelled_camera.undistort_pixels_classical(user_annotation[:2],
                                                                                cam_labelled_dist_coeff)
            cam_1_ray = labelled_camera(undistorted_annotations)
            origin = cam_1_ray.origin[:,0][:,None]
            direction = cam_1_ray.direction[:,0][:,None]
            s = torch.linspace(prism_distance, prism_distance + 10, 50).to(torch.float64)
            r = origin + s[None, :] * direction        
            
            annotations_curve_unlabelled_camera = unlabelled_camera.reproject(torch.tensor(r).to(torch.float64), R_unlabelled, T_unlabelled)            
            annotations_curve_unlabelled_camera = unlabelled_camera.distort_pixels_classical(
                                                                annotations_curve_unlabelled_camera,
                                                                cam_unlabelled_dist_coeff)
            return annotations_curve_unlabelled_camera

def get_secondary_camera(arena):
    """
    Returns secondary camera given the arena model parameters
    """
    R_stereo_cam = get_rot_mat(
            arena.stereo_camera_angles[0],
            arena.stereo_camera_angles[1],
            arena.stereo_camera_angles[2],
            )
    return arena.get_stereo_camera(arena.principal_point_pixel_cam_1,
                                arena.focal_length_cam_1,
                                R_stereo_cam,
                                arena.T_stereo_cam,
                                r1=arena.stereocam_r1,
                                radial_dist_coeffs=arena.radial_dist_coeffs_cam_1)
    

# cam_label is the label of the camera whose annotation is provided by the user
if cam_label == "primary":
    camera_virtual = arena.camera1
    cam_virtual_dist_coeff = arena.radial_dist_coeffs_cam_0
    camera_real = get_secondary_camera(arena)
    cam_real_dist_coeff = arena.radial_dist_coeffs_cam_1

elif cam_label == "secondary":
    camera_virtual = get_secondary_camera(arena)
    cam_virtual_dist_coeff = arena.radial_dist_coeffs_cam_1
    camera_real = arena.camera1
    cam_real_dist_coeff = arena.radial_dist_coeffs_cam_0
else:
    print("Please provide appropriate camera labels")

user_annotation = torch.tensor(user_annotation)
epipolar_line_real, epipolar_line_virtual = get_annotations_curve(arena, user_annotation, 
                                      camera_real, cam_real_dist_coeff, 
                                      camera_virtual, cam_virtual_dist_coeff)
if "virtual" in cam_label:
    epipolar_line_unlabelled, epipolar_line_labelled = get_epipolar_line(arena, user_annotation, cam_label)
    epipolar_line_unlabelled = epipolar_line_unlabelled.numpy()
    epipolar_line_labelled = epipolar_line_labelled.numpy()
elif "real" in cam_label:
    epipolar_line_unlabelled = get_epipolar_line(arena, user_annotation, cam_label)
    epipolar_line_unlabelled = epipolar_line_unlabelled.numpy()

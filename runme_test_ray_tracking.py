# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pytorch_local
#     language: python
#     name: python3
# ---

# %% Imports
from ray_tracing_simulator import Prism, Ray, Plane, OpticalPlane, Camera, visualize_camera_configuration, closest_point
import matplotlib.pyplot as plt
import numpy as np  
import torch
import scipy.io as sio
import os
pi = torch.tensor(np.pi)


# %% Testing class Plane
# Single plane
plane = Plane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[1.,0.,0.], a=1.,b=1.)
_, ax = plane.visualize()
ax.set_aspect('equal', adjustable='datalim')   


# %% Testing class Ray
# Single ray
ray = Ray(origin=[0.,0.,0.], direction=[1.,0.,0.])
_, ax = ray.visualize()
ax.set_aspect('equal', adjustable='datalim')   


# %% Testing class OpticalPlane with a single ray refraction
# Optical plane refraction
plane = OpticalPlane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[0.,0.,0.], a=1., b=1., 
              refractive_idx_1=1., refractive_idx_2=1.5)
incident_ray = Ray(origin=[-0.5,0.,0.], direction=[1.,0.,0.])
refracted_ray = plane.refract_ray(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = refracted_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 


# %% Testing class OpticalPlane with a single ray reflection
# Optical plane reflection
plane = OpticalPlane(alpha=np.pi/3, beta=np.pi/6, gamma=np.pi/10, center=[0.,0.,0.], a=1.,b=1., 
              refractive_idx_1=1., refractive_idx_2=1.5)
incident_ray = Ray(origin=[-0.5,0.,0.], direction=[1.,0.,0.])
reflected_ray = plane.reflect_ray(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = reflected_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 


# %% Testing class Prism
# Testing a single ray refracting and reflecting through a prism
n_glass = 1.55
n_air = 1.
prism_alpha = 0.
prism_beta = pi / 2
prism_gamma = pi / 2 
prism_center = np.array([0.,0.,0.])[:, None]
prism = Prism(prism_size=[1.,1.,1.], prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                prism_center=prism_center, refractive_index_glass=n_glass, 
                refractive_index_air=n_air)
origin_point=[0.,0.6,-0.25] 
target_point=[0.,0.37,0.]
prism.trace_ray(origin_point, target_point) 
fig, ax = prism.visualize_prism_and_ray()
ax.set_aspect('equal', adjustable='datalim')        
plt.show()


# %% Test single camera configuration with prism and rays
pixels = torch.rand(2, 10)
height = 1200
width = 1918
pixels = pixels * torch.tensor([width, height])[:, None]
_, ax, prism, camera = visualize_camera_configuration(pixels=pixels)
ray_direct = camera.initialize_ray(pixels)
points = closest_point(prism.ray4, ray_direct)


# %% Visualize camera configuration
_, ax = plt.subplots(1, 2)
fig, ax[0], _, _ = visualize_camera_configuration(pixels=pixels, ax=ax[0])
prism_lims = [prism.prism_center - torch.tensor(prism.prism_size).unsqueeze(-1), 
                prism.prism_center + torch.tensor(prism.prism_size).unsqueeze(-1)]
ax[0].set_xlim(prism_lims[0][0], prism_lims[1][0])
ax[0].set_ylim(prism_lims[0][1], prism_lims[1][1])
ax[0].set_zlim(prism_lims[0][2], prism_lims[1][2])
cam_lims = [camera.center - 1, camera.center + 1]
_, ax[1], _, _ = visualize_camera_configuration(pixels=pixels, ax=ax[1])
ax[1].set_xlim(cam_lims[0][0], cam_lims[1][0])
ax[1].set_ylim(cam_lims[0][1], cam_lims[1][1])
ax[1].set_zlim(cam_lims[0][2], cam_lims[1][2])


# %% Validate real camera rays: Calculate distance of the ray from ground truth 3-D points
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
mat = sio.loadmat(calibration_results_path)
undistorted_real_pixels = torch.tensor(mat['output_data_cam_0_undistorted'], dtype=torch.float32).T
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T
test_idx = torch.randperm(undistorted_real_pixels.shape[1])
principal_point_pixel = [638.040, 492.499] # This comes from the calibration results
camera = Camera(principal_point_pixel=principal_point_pixel)
undistorted_real_pixels = undistorted_real_pixels[:, test_idx]
target_coordinates = target_coordinates[:, test_idx]
print(f'Finished loading calibration results from {calibration_results_path}')
ray_direct = camera.initialize_ray(undistorted_real_pixels)
distance = ray_direct.distance_to_point(target_coordinates)


# %% Visualize 'n_sample' rays traced from pixels
num_samples = 10
test_idx = torch.randperm(undistorted_real_pixels.shape[1])[:num_samples]
principal_point_pixel=[638.040, 492.499]
camera = Camera(principal_point_pixel=principal_point_pixel)
undistorted_real_pixels = undistorted_real_pixels[:, test_idx]
target_coordinates = target_coordinates[:, test_idx]
ray_direct = camera.initialize_ray(undistorted_real_pixels)
distance = ray_direct.distance_to_point(target_coordinates)
fig, ax = camera.visualize()
ray_direct.t *= 160
ray_direct.visualize(fig=fig, ax=ax)
ax.scatter(target_coordinates[0], target_coordinates[1], target_coordinates[2], c='r')


# %%  Visualize two cameras, given the rotation and translation matrix of one with respect to the other
R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None]
camera1 = Camera(principal_point_pixel=[638.040, 492.499])
camera2 = Camera(principal_point_pixel=[659.37, 521.507], focal_length=19.69)
camera2.update_camera_pose(R, T)
fig, ax = camera1.visualize()
camera2.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 


# %% Visualize rays from two cameras
num_samples = 10
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
mat = sio.loadmat(calibration_results_path)
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_undistorted'], dtype=torch.float32).T
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_undistorted'], dtype=torch.float32).T
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T
test_idx = torch.randperm(undistorted_real_pixels_cam_0.shape[1])[:num_samples]
principal_point_pixel_cam_0 = [638.040, 492.499] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.37, 521.507] # This comes from the calibration results
camera1 = Camera(principal_point_pixel=principal_point_pixel_cam_0)
camera2 = Camera(principal_point_pixel=principal_point_pixel_cam_1, focal_length=19.69)
camera2.update_camera_pose(R, T)
undistorted_real_pixels_cam_0 = undistorted_real_pixels_cam_0[:, test_idx]
undistorted_real_pixels_cam_1 = undistorted_real_pixels_cam_1[:, test_idx]

target_coordinates = target_coordinates[:, test_idx]
print(f'Finished loading calibration results from {calibration_results_path}')
ray_direct_1 = camera1.initialize_ray(undistorted_real_pixels_cam_0)
ray_direct_1.t *= 160
ray_direct_2 = camera2.initialize_ray(undistorted_real_pixels_cam_1)
ray_direct_2.t *= 160
fig, ax = camera1.visualize()
fig, ax = camera2.visualize(fig=fig, ax=ax)
fig, ax = ray_direct_1.visualize(fig=fig, ax=ax)
fig, ax = ray_direct_2.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim')


# %%
calibration_results_dir = '/groups/branson/bransonlab/aniket/fly_walk_imaging/prism/exp_18/results-non-corroded/'
calibration_results_file = 'ball_bearing_data.mat'
calibration_results_path = os.path.join(calibration_results_dir, calibration_results_file)
mat = sio.loadmat(calibration_results_path)
undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_undistorted'], dtype=torch.float32).T - 1
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_undistorted'], dtype=torch.float32).T - 1
target_coordinates = torch.tensor(mat['input_data'], dtype=torch.float32).T
test_idx = torch.randperm(undistorted_real_pixels_cam_0.shape[1])
principal_point_pixel_cam_0 = [638.040 - 1, 492.499 - 1] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778 - 1, 521.5078 - 1] # This comes from the calibration results
camera1 = Camera(principal_point_pixel=principal_point_pixel_cam_0)
camera2 = Camera(principal_point_pixel=principal_point_pixel_cam_1, focal_length=19.697)
camera2.update_camera_pose(R, T)
undistorted_real_pixels_cam_0 = undistorted_real_pixels_cam_0[:, test_idx]
undistorted_real_pixels_cam_1 = undistorted_real_pixels_cam_1[:, test_idx]

target_coordinates = target_coordinates[:, test_idx]
print(f'Finished loading calibration results from {calibration_results_path}')
ray_direct_1 = camera1.initialize_ray(undistorted_real_pixels_cam_0)
ray_direct_2 = camera2.initialize_ray(undistorted_real_pixels_cam_1)
distance1 = ray_direct_1.distance_to_point(target_coordinates)
distance2 = ray_direct_2.distance_to_point(target_coordinates)
# Closest approach
recon_3D, lines_distance = closest_point(ray_direct_1, ray_direct_2)
recon_3D_error = torch.norm(recon_3D - target_coordinates, dim=0)
print(f'Mean distance error for camera 1: {distance1.mean()}')
print(f'Mean distance error for camera 2: {distance2.mean()}')
print(f'Mean distance error for closest approach: {recon_3D_error.mean()}')

# %%

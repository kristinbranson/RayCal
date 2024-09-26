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
from ray_tracing_simulator_nnModules_grad import Prism, Ray, Plane, ReflectingPlane, RefractingPlane, Camera, Arena, visualize_camera_configuration
import matplotlib.pyplot as plt
import numpy as np  
import torch
import scipy.io as sio
import os
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
#from torchsummary import summary
torch.autograd.set_detect_anomaly(True)
pi = torch.tensor(np.pi)
test_output_folder = 'test_outputs'
os.makedirs(test_output_folder, exist_ok=True)

# %% Return angle of incidence and emergence for each plane in the prism, given input rays
def calculate_angle_of_incidence(incident_ray, prism):
    plane1, plane2, plane3 = prism.get_planes(prism.prism_center, prism.prism_angles)
    ray1, ray2, ray3 = prism(incident_ray)
    refraction_angle_i = torch.acos(torch.matmul(incident_ray.direction.T, -plane1.axes[:,0][:,None]))
    refraction_angle_o = torch.acos(torch.matmul(ray1.direction.T, -plane1.axes[:,0][:,None]))
    snells_law_angle = torch.asin(
                            prism.refractive_index_air / prism.refractive_index_glass * torch.sin(refraction_angle_i)
                            )
    ray_1_angles_bundle = torch.stack([refraction_angle_i*180/pi, 
        refraction_angle_o*180/pi, snells_law_angle*180/pi]
        ).detach().numpy()

    reflection_angle_i = torch.acos(torch.matmul(ray1.direction.T, plane2.axes[:,0][:,None]))
    reflection_angle_o = torch.acos(torch.matmul(ray2.direction.T, -plane2.axes[:,0][:,None]))
    ray_2_angles_bundle = torch.stack([reflection_angle_i*180/pi, 
    reflection_angle_o*180/pi]).detach().numpy()

    refraction_angle_i = torch.acos(torch.matmul(ray2.direction.T, plane3.axes[:,0][:,None]))
    refraction_angle_o = torch.acos(torch.matmul(ray3.direction.T, plane3.axes[:,0][:,None]))
    snells_law_angle = torch.asin(
        prism.refractive_index_glass / prism.refractive_index_air * torch.sin(refraction_angle_i))
    ray_3_angles_bundle = torch.stack([refraction_angle_i*180/pi, 
    refraction_angle_o*180/pi, 
    snells_law_angle*180/pi]).detach().numpy()

    return ray_1_angles_bundle, ray_2_angles_bundle, ray_3_angles_bundle

def plot_two_camera_figures(ang1, ang2):
    """
    ang1 is the set of angles for camera 1
    ang2 is the set of angles for camera 2
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # Plot a straight line    
    ax[0].scatter(ang1[0,:,0], ang1[1,:,0], s=0.75, color='red', label='Data')
    ax[0].set_xlabel('Angle of incidence ($^o$)', fontsize=25)
    ax[0].set_ylabel('Angle of refraction ($^o$)', fontsize=25)
    ax[0].tick_params(axis='x', labelsize=15)
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].set_title('Camera 1', fontsize=25)    
    x = np.linspace(ang1[0,:,0].min(), ang1[0,:,0].max(), 100)
    y = x
    ax[0].plot(x, y, 'r--', alpha=0.8, color='black', label='y=x')
    ax[0].legend(fontsize=15, loc='upper left')
      
    ax[1].scatter(ang2[0,:,0], ang2[1,:,0], s=0.75, color='red', label='Data')
    ax[1].set_xlabel('Angle of incidence ($^o$)', fontsize=25)
    ax[1].set_ylabel('Angle of refraction ($^o$)', fontsize=25)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].set_title('Camera 2', fontsize=25)
    # Plot a straight line  
    x = np.linspace(ang2[0,:,0].min(), ang2[0,:,0].max(), 100)
    y = x
    ax[1].plot(x, y, 'r--', alpha=0.8, color='black', label='y=x')
    ax[1].legend(fontsize=15, loc='upper left')
    return fig, ax
    

# %% Testing class Plane
# Single plane
plane = Plane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[1.,0.,0.], a=1., b=1.)
_, ax = plane.visualize()
ax.set_aspect('equal', adjustable='datalim')   


# %% Testing class Ray
# Single ray
ray = Ray(origin=torch.tensor([0.,0.,0.])[:,None], direction=torch.tensor([1.,0.,0.])[:,None])
_, ax = ray.visualize()
ax.set_aspect('equal', adjustable='datalim')   


# %% Testing class OpticalPlane with a single ray refraction
# Optical plane refraction
plane = RefractingPlane(alpha=pi/3, beta=pi/6, gamma=pi/10, center=[0.,0.,0.], a=1., b=1., 
              refractive_idx_1=1., refractive_idx_2=1.5)
incident_ray = Ray(origin=torch.tensor([-0.5,0.,0.])[:,None], direction=torch.tensor([1.,0.,0.])[:,None])
refracted_ray = plane(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = refracted_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 


# %% Testing class OpticalPlane with a single ray reflection
# Optical plane reflection
plane = ReflectingPlane(alpha=np.pi/3, beta=np.pi/6, gamma=np.pi/10, center=[0.,0.,0.], a=1.,b=1.)
incident_ray = Ray(origin=torch.tensor([-0.5,0.,0.])[:,None], direction=torch.tensor([1.,0.,0.])[:,None])
reflected_ray = plane(incident_ray)
fig, ax = plane.visualize()
fig, ax = incident_ray.visualize(fig=fig, ax=ax)
_, ax = reflected_ray.visualize(fig=fig, ax=ax)
ax.set_aspect('equal', adjustable='datalim') 


# %% Testing class Prism
# Testing a single ray refracting and reflecting through a prism
n_glass = 1.55
n_air = 1.
prism_alpha = 0.
prism_beta = 0.
prism_gamma = 0.
prism_center = np.array([0.,0.,0.])[:, None]
prism = Prism(prism_size=[1.,1.,1.], prism_angles=[prism_alpha, prism_beta, prism_gamma], 
                prism_center=prism_center, refractive_index_glass=n_glass, 
                refractive_index_air=n_air)
origin_point=torch.tensor([0.,0.6,-0.25])[:,None]
target_point=torch.tensor([0.,0.37,0.])[:,None]
ray = Ray(origin=origin_point, target=target_point)
prism(ray)
fig, ax = prism.visualize_prism_and_ray(ray)
ax.set_aspect('equal', adjustable='datalim')
plt.show()


# %% Test single camera configuration with prism and rays
pixels = torch.rand(2, 10)
height = 1200
width = 1920
pixels = pixels * torch.tensor([width, height])[:, None]
_, ax, prism, camera = visualize_camera_configuration(pixels=pixels)
ray_direct = camera.initialize_ray(pixels)
#points = closest_point(prism.ray4, ray_direct)


# %% Visualize camera configuration
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
#_, ax = plt.subplots(1, 2, projection='3d')
fig, ax1, _, _ = visualize_camera_configuration(pixels=pixels, fig=fig, ax=ax1)
prism_lims = [prism.prism_center - torch.tensor(prism.prism_size).unsqueeze(-1), 
                prism.prism_center + torch.tensor(prism.prism_size).unsqueeze(-1)]
ax1.set_xlim(prism_lims[0][0], prism_lims[1][0])
ax1.set_ylim(prism_lims[0][1], prism_lims[1][1])
ax1.set_zlim(prism_lims[0][2], prism_lims[1][2])
cam_lims = [camera.center.detach().numpy() - 5, 
            camera.center.detach().numpy() + 5]
_, ax2, _, _ = visualize_camera_configuration(pixels=pixels, fig=fig, ax=ax2)
ax2.set_xlim(cam_lims[0][0], cam_lims[1][0])
ax2.set_ylim(cam_lims[0][1], cam_lims[1][1])
ax2.set_zlim(cam_lims[0][2], cam_lims[1][2])


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
print(f'Mean distance error: {distance.mean()}')

# %% Visualize 'n_sample' rays traced from pixels (single camera)
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
ax.set_title(f'{num_samples} rays traced from ball bearing centroid projections', fontsize=15)


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
ax.set_title('Two cameras with a given relative pose', fontsize=15)


# %% Visualize {num_samples} rays from two cameras
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
ax.set_title(f'{num_samples} rays from ball bearing centroid projections on two cameras', fontsize=15)


# %% Calculate errors for two cameras
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

# %% Testing Arena
principal_point_pixel_cam_0 = [638.040 - 1, 492.499 - 1] # This comes from the calibration results
principal_point_pixel_cam_1 = [659.3778 - 1, 521.5078 - 1]

R = torch.tensor([[0.819301743677432, 0.0073199538315673, -0.573315856298274],
                   [-1.41589415524092e-05, 0.999918760232094, 0.0127464793349662], 
                   [0.573362583891418, -0.0104350951991858, 0.819235287436729]]).T
T = torch.tensor([72.8566307938209, -0.980908710814855, 22.7386226749512])[:, None]

focal_length_cam_1 = 19.65
focal_length_cam_2 = 19.69

prism_distance = 130.
arena = Arena(principal_point_pixel_cam_0, 
principal_point_pixel_cam_1, 
focal_length_cam_1, 
focal_length_cam_2, 
R, 
T, 
prism_distance,
prism_angles = torch.tensor([0., 0., 0.]))


# %% Check if refraction works as intended
import matplotlib.pyplot as plt

undistorted_real_pixels_cam_0 = torch.tensor(mat['output_data_cam_0_undistorted'], dtype=torch.float32).T - 1
undistorted_real_pixels_cam_1 = torch.tensor(mat['output_data_cam_1_undistorted'], dtype=torch.float32).T - 1
camera1 = Camera(principal_point_pixel=principal_point_pixel_cam_0)
camera2 = Camera(principal_point_pixel=principal_point_pixel_cam_1, focal_length=19.697)
camera2.update_camera_pose(R, T)
ray1 = camera1(undistorted_real_pixels_cam_0)
ray2 = camera2(undistorted_real_pixels_cam_1)
ang11, ang12, ang13 = calculate_angle_of_incidence(ray1, arena.prism)
ang21, ang22, ang23 = calculate_angle_of_incidence(ray2, arena.prism)
fig, ax = plot_two_camera_figures(ang12, ang22)
ax[0].set_xlabel('Angle of incidence ($^o$)')
ax[1].set_xlabel('Angle of reflection ($^o$)')
fig.suptitle('Angle of reflection vs Angle of incidence', fontsize=25)

fig.savefig(f'{test_output_folder}/reflection.png')
plot_two_camera_figures(ang11[1:,...], ang21[1:,...])
ax[0].set_xlabel('Angle of refraction ($^o$)')
ax[1].set_xlabel("Snell's law estimate ($^o$)")
fig.suptitle('Angle of refraction vs Snells Law (First Plane)', fontsize=25)
fig.savefig(f'{test_output_folder}/refraction_plane1.png')

plot_two_camera_figures(ang13[1:,...], ang23[1:,...])
fig.suptitle('Angle of refraction vs Snells Law (Third Plane)', fontsize=25)
ax[0].set_xlabel('Angle of refraction ($^o$)')
ax[1].set_xlabel("Snell's law estimate ($^o$)")
fig.savefig(f'{test_output_folder}/refraction_plane3.png')


# ## Check if reflection works as intended


# %%

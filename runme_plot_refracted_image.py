"""
All distances are in cm
"""

import cv2 as cv
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import pickle
import matplotlib.pyplot as plt
from config import Config

def refraction_model(y0):
    y = prism_camera_distance * (object_point - y0) / prism_object_distance
    return np.abs(y / np.sqrt(prism_camera_distance ** 2 + y ** 2) - n_glass_air * (y0 - y) / np.sqrt(prism_thickness ** 2 + (y0 - y) ** 2))

def get_initial_guess(object_point):
    min_error = 1e6
    for y0 in np.linspace(0, 50 * object_height, 100 * num_points):
        if np.abs(refraction_model(y0)) < min_error:
            min_error = np.abs(refraction_model(y0))
            best_y0 = y0
    return best_y0

def convert_world_to_pixel_coordinates(world_x, world_y, camera_rotation):
    """
    world_x: x coordinate of the point on the camera sensor in the world reference frame
    world_y: y coordinate of the point on the camera sensor in the world reference frame
    camera_rotation: angle of rotation of the camera from the world reference frame
    returns: distance of the point from the center of the camera sensor in the pixel reference frame
    """
    camera_center_x = -focal_length * np.cos(camera_rotation)
    camera_center_y = -focal_length * np.sin(camera_rotation)
    camera_projection_direction_unit_vector = np.array([0, 1])
    rot_mat = np.array([[np.cos(camera_rotation), -np.sin(camera_rotation)], [np.sin(camera_rotation), np.cos(camera_rotation)]])
    camera_projection_direction_unit_vector = rot_mat @ camera_projection_direction_unit_vector
    image_projection_vector = np.array([world_x[0], world_y[0]]) - np.array([camera_center_x, camera_center_y])
    return pixel_scaling_factor * (image_projection_vector @ camera_projection_direction_unit_vector)
    
def find_intersection_with_camera_sensor(y0_solution_, focal_length_, camera_rotation_, prism_camera_distance_, object_point_, prism_object_distance_):
    y = prism_camera_distance * (object_point_ - y0_solution_) / prism_object_distance_
    camera_intersection_x = -focal_length_ / (np.cos(camera_rotation_) + np.sin(camera_rotation_) * y / prism_camera_distance_) 
    camera_intersection_y = camera_intersection_x * y / prism_camera_distance_
    return camera_intersection_x, camera_intersection_y

num_points = Config.num_points
refractive_index_glass = Config.refractive_index_glass # N-BK7 glasss
refractive_index_air = Config.refractive_index_air
n_glass_air = refractive_index_glass / refractive_index_air
prism_object_distance = Config.prism_object_distance
prism_camera_distance = Config.prism_camera_distance
prism_thickness = Config.prism_thickness
focal_length = Config.focal_length
object_camera_distance = prism_object_distance + prism_thickness + prism_camera_distance
object_height = Config.object_height
pixel_scaling_factor = 1
camera_rotation = Config.theta_camera # rotated downwards is positive

with open('/groups/branson/bransonlab/aniket/fly_walk_imaging/calibration_code/refraction_model/data/simulation_coordinates.pkl', 'rb') as f:
    data = pickle.load(f)
object_points = np.linspace(-object_height, object_height, num_points)
object_points = data[1][:,1]
y0_kristins_simulation = data[2][:,1]
image_points_kristins_simulation_world = data[0]
image_points_kristins_simulation_pixels = np.linspace(-Config.image_height, 0, num_points) * pixel_scaling_factor
image_points_pixels = np.zeros(len(object_points))
image_points_world = np.zeros((len(object_points), 2))

for point_id, object_point in enumerate(object_points):
    y0_initial_guess = get_initial_guess(object_point)
    y0_solution = least_squares(refraction_model, y0_initial_guess).x    
    camera_intersection_x, camera_intersection_y = find_intersection_with_camera_sensor(
    y0_solution, focal_length, camera_rotation, prism_camera_distance, object_point, prism_object_distance)
    image_points_world[point_id,:] = np.array([camera_intersection_x, camera_intersection_y])[:,0]
    image_points_pixels[point_id] = convert_world_to_pixel_coordinates(camera_intersection_x, camera_intersection_y, camera_rotation)
    


fig, ax = plt.subplots(3,1,figsize=(8,13))
colors = plt.get_cmap('jet')(np.linspace(0, 1.0, num_points))
for i in range(num_points):
    ax[0].plot(i, image_points_pixels[i],'o',color=colors[i])
ax[0].plot(image_points_pixels, color='black')
ax[0].set_xlabel('Image points id', fontsize = 24)
ax[0].set_ylabel('Image points distance', fontsize=24)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].tick_params(axis='y', labelsize=18)

for i in range(num_points):
    ax[1].plot(i, object_points[i],'o',color=colors[i])
ax[1].plot(object_points, color='black')
ax[1].set_xlabel('Object points id', fontsize = 24)
ax[1].set_ylabel('Object points distance', fontsize = 24)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)

for i in range(num_points):
    ax[2].plot(image_points_pixels[i], object_points[i],'o',color=colors[i])
ax[2].plot(image_points_pixels, object_points, color='black')
ax[2].set_xlabel('Image points distance', fontsize = 24)
ax[2].set_ylabel('Object points distance', fontsize = 24)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].tick_params(axis='y', labelsize=18)

plt.figure(figsize=(10,10))
for i in range(num_points):
    plt.plot(image_points_world[i,0], image_points_world[i,1], 'o', color=colors[i], markersize=3)
    plt.plot(image_points_kristins_simulation_world[i,0], image_points_kristins_simulation_world[i,1], 'x', color=colors[i])
plt.xlabel('x', fontsize=24)
plt.ylabel('y', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['Forward', 'Backward'])